# drangue

An agent runtime for Python.

An agent is just a model plus tools. Running it is one call. No graphs, no
chains, no base classes to inherit. You can read the whole loop in one sitting.

## Install

The core has zero dependencies. Install the adapter for the backend you want:

```bash
pip install "drangue[openai]"     # OpenAI, DeepSeek, Groq, Ollama, and more
pip install "drangue[anthropic]"  # Claude
```

## The whole thing

```python
from drangue import Agent, tool

@tool
def error_rate(service: str) -> str:
    """Return the recent 5xx error rate and p99 latency for a service."""
    # In production this queries Prometheus/Datadog; here it returns a sample.
    return f"{service}: 4.2% 5xx over 15m (baseline 0.3%), p99 latency 1.8s"

agent = Agent(
    model="claude-opus-4-8",
    tools=[error_rate],
    instructions="You are an on-call assistant. Be concise and specific.",
)

result = agent.run_sync("Is the checkout service healthy right now?")
print(result.output)
```

That is the entire surface for the happy path. A tool is a typed function.
The decorator reads its signature and docstring and builds the schema for you.
No manual JSON, no Pydantic required (though you can pass plain functions too).

The core is async. `run_sync` is the convenience wrapper for scripts; inside
async code you `await` instead:

```python
result = await agent.run("Is the checkout service healthy right now?")
```

## See every step

Inspection is one flag, not a separate service:

```python
result = agent.run_sync("Is the checkout service healthy right now?", trace=True)
```

```
* tool   error_rate(service='checkout')
       -> checkout: 4.2% 5xx over 15m (baseline 0.3%), p99 latency 1.8s
* model  checkout is unhealthy: 4.2% 5xx is 14x the 0.3% baseline, p99 at 1.8s. Worth paging.
```

`result.usage` reports the token totals for the run, and `result.events` is the
full event log the run was driven from.

## Drive the loop yourself

`stream` yields each event as it is appended to the log, plus live token
chunks (`model_delta`) while the model is still talking, so you stay in
control:

```python
async for event in agent.stream("Is the orders service healthy?"):
    if event.type == "model_delta":
        print(event.payload["text"], end="", flush=True)   # live tokens
    elif event.type == "model_decision":
        for call in event.payload["tool_calls"]:
            print("calling", call["name"], call["arguments"])
    elif event.type == "run_finished":
        print(event.payload["output"])
```

Deltas are ephemeral: they stream to you but are never stored, so the recorded
log stays the source of truth and replays are unchanged.

## Conversations

A completed run accepts follow-up turns: same `run_id`, new input. Each turn
is a recorded `user_message` event, so a conversation resumes after a crash
and replays deterministically like any other run:

```python
first = await agent.run("what is 2+2?", run_id="chat-7")
second = await agent.run("and double that?", run_id="chat-7")   # sees turn one
```

A run that is mid-flight or paused refuses a new input; repeating the latest
turn's input replays it instead of duplicating it.

## Structured output

Constrain the final answer to a JSON schema. The model delivers through a
synthetic `final_answer` tool; the runtime validates, records, and retries a
nonconforming answer with the errors spelled out:

```python
schema = {"type": "object", "required": ["answer"],
          "properties": {"answer": {"type": "integer"}}}

result = await agent.run("what is 2+3?", output_schema=schema)
result.output_parsed   # {"answer": 5}, validated
```

The schema is recorded at run start, so a resumed run keeps the original
contract. A run that exhausts its corrections finishes with plain text and
`output_parsed` is None, so a miss is visible instead of guessed at.

## MCP servers as tool packs

Any Model Context Protocol server's tools become ordinary drangue tools with
`pip install drangue-mcp`; guardrails, autonomy, and the event log apply to
them unchanged:

```python
from drangue_mcp import stdio_toolset

async with stdio_toolset("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]) as toolset:
    agent = Agent("claude-opus-4-8", tools=toolset.tools)
    result = await agent.run("what is in /data?")
```

## Cost control

Cap a run's spend and it stops gracefully before an unaffordable step, using the
token usage recorded in the log:

```python
from drangue import Agent, Budget

agent = Agent("claude-opus-4-8", tools=tools, budget=Budget(max_tokens=200_000))
# or a dollar budget with a price table:
prices = {"claude-opus-4-8": {"input": 15.0, "output": 75.0}}   # $ per 1M tokens
agent = Agent("claude-opus-4-8", tools=tools,
              budget=Budget(max_usd=0.50, prices=prices))

result = await agent.run("...")
result.usage        # {"input_tokens": ..., "output_tokens": ...}
result.cost(prices) # dollars, priced per step against the model that ran
```

Prices are yours to supply: they vary by provider and drift, so the library
ships no table and will not guess. That cuts both ways: `Budget(max_usd=...)`
without `prices` raises rather than quietly computing $0 and never firing, and
a run that reaches a model the table does not cover stops with a recorded
"budget unenforceable" instead of counting it as free. Cached prompt tokens are
counted and priced too (`cache_write` / `cache_read` price keys, defaulting to
Anthropic's 1.25x / 0.1x ratios), so `cache=True` does not hide spend from the
limit.

Route each step to the cheapest model that can handle it. The model that
actually ran is recorded per step, so routing is visible in the trace and
counted in the budget:

```python
from drangue import Agent, RuleRouter

router = RuleRouter(
    default=cheap_model,
    rules=[(lambda messages, i: i == 0, smart_model)],   # only the first step is judgment
)
agent = Agent(model=router, tools=tools)
```

For repeated runs, `Agent("claude-opus-4-8", tools=tools, cache=True)` marks the
stable prefix (system prompt and tool definitions) for prompt caching. Context
is already ordered stable-to-volatile, so the cacheable part stays at the front.
If you build the model yourself, set it there instead
(`AnthropicModel("claude-opus-4-8", cache=True)`), since that model, not the
facade, owns the setting.

## Resilient tools

Tools are bounded by default and never crash a run: an exception comes back to
the model as a clean, structured failure it can reason about. Opt into more with
options on `@tool`:

```python
from drangue import tool, RateLimitError

@tool(timeout=5.0, retries=3, backoff=0.5)
def fetch_metrics(service: str) -> str:
    """Fetch metrics, retried on transient failures."""
    resp = http_get(service)
    if resp.status == 429:
        raise RateLimitError(retry_after=resp.headers["Retry-After"])  # retried, honoring the hint
    return resp.text
```

Independent sibling calls run concurrently when it is safe (every call
autonomous, every tool reversible) and serially otherwise, with results
recorded in call order so replay is deterministic either way.

The wrapper applies, in order: timeout, classify the failure, retry transient
ones with jittered exponential backoff (reusing the idempotency key), validate
the result, then return a clean failure or a marked-degraded `fallback`.
Timeouts are NOT retried by default: a timed-out sync tool may still be
running, so retrying could execute a side effect twice; opt in with
`retry_on=DEFAULT_RETRY_ON + (TimeoutError,)` where that is safe. A
server-sent `Retry-After` is honored as given, never shortened. The model
receives, for example:

```json
{"ok": false, "tool": "fetch_metrics", "error": {"category": "timeout", "message": "timeout"}}
```

## Guardrails

Constrain what the agent can do, enforced in code regardless of what the model
decides (a prompt instruction is the thing injection overrides). A blocked call
comes back to the model as a clean failure it can reason about.

```python
from drangue import Agent, Guardrails

guard = Guardrails(
    allow={"read_metrics", "search"},        # the agent is a constrained principal
    require_approval_for_irreversible=True,   # gate the dangerous ones
    approver=lambda name, args: ask_human(name, args),
    input_guard=lambda text: "blocked" if looks_malicious(text) else None,
    output_guard=lambda name, args: detect_exfiltration(name, args),
    result_guard=lambda name, text: flag_injection(text),   # screen what tools RETURN
)
agent = Agent("claude-opus-4-8", tools=tools, guardrails=guard)
```

Mark a tool's stakes so gates can act on them:

```python
@tool(reversible=False, requires_approval=True)
def delete_database(name: str) -> str:
    """Delete a database (irreversible, always gated)."""
    ...
```

The layers are independent: an allow-list bounds reach, action gates stop the
irreversible, and the input, output, and result guards catch malicious content
on the way in, suspicious actions on the way out, and poisoned tool results
(the classic injection carrier) before the model reads them. No single layer is
sufficient; together they make a successful injection survivable.

## Evals and deploy gates

Score the agent statistically across correctness, safety, and efficiency, then
gate deploys on real regressions. A run is repeated several times (agents are
non-deterministic) and produces a profile, not a pass/fail.

```python
from drangue import Agent, Scenario, Gate, evaluate, output_contains, forbids_tool

scenarios = [
    Scenario("answers", "what is 2+3?", checks=[output_contains("5")], runs=5),
    Scenario("safety", "clean up the database",
             checks=[forbids_tool("delete_all")], runs=5),
]

baseline = (await evaluate(deployed_agent, scenarios)).profile()
candidate = (await evaluate(new_agent, scenarios)).profile()

decision = Gate().evaluate(baseline, candidate)
if not decision.passed:
    raise SystemExit(f"deploy blocked: {decision.blocks}")
```

Safety is exact set membership (a rule, not a judge); open-ended correctness can
use an `LLM Judge`. Every rate comes with the standard error that says how solid
it is (`profile()["noise"]`), and the gate warns when its own threshold sits
inside that band, because a 5% rule cannot resolve a difference measured to
±25%. The gate compares against the baseline, blocks on safety and on
correctness past a noise band, warns on efficiency, and makes an override state
a reason:

```python
decision.override("hotfix for outage", by="oncall@team", record=audit_log.append)
```

Turn a traced production failure into a regression scenario with
`scenario_from_result(result, name, checks=...)`, so the eval set grows from what
actually went wrong.

## Human in the loop

Autonomy is granted per action, not per agent. Each tool runs in one of three
modes: `shadow` (propose, do not execute), `assisted` (pause for a human), or
`autonomous` (execute, review later).

```python
from drangue import Agent, Autonomy, SQLiteStore

agent = Agent("claude-opus-4-8", tools=tools, store=SQLiteStore("runs.db"),
              autonomy=Autonomy(default="autonomous", modes={"wire_funds": "assisted"}))

result = await agent.run("pay the invoice", run_id="pay-1")
if result.status == "paused":
    for p in result.pending_approvals:
        print(p["tool"], p["arguments"], "because:", p["reasoning"])  # the case, not a bare action
    await agent.approve("pay-1")        # or agent.reject("pay-1", reason="...")
    result = await agent.resume("pay-1")
```

An assisted action is a durable pause: the approval request and the human's
decision are events in the log, so a paused run survives a process restart and
resumes by replay. The side effect happens once, only after approval.

## Durable runs

Point an Agent at a durable store and give a run a stable `run_id`. If the
process dies mid-run, a new one resumes from exactly where it stopped: recorded
steps are replayed as facts, so recorded work is never redone. The precise
guarantees, stated honestly: a recorded step never re-executes; a tool that
declares `idempotency_key` deduplicates even a crash that lands between the
side effect and its record; a tool marked `reversible=False` gets an intent
marker before it runs, so that crash window is *detected* on resume (the model
receives a clean `unknown_outcome` failure) instead of re-executed; plain
reversible tools and model calls are at-least-once.

Two more properties ride on the log. The run's config (max_steps, autonomy
modes, the tool set) is recorded at start, so a resumed run decides as the
original was configured, not as the agent happens to be configured today. And
stores that support leasing (both built-ins and drangue-postgres do) allow one
live driver per run: a second process gets a clear `LeaseHeldError`, while a
dead process's lease lapses on its TTL so crash recovery needs no cleanup.

```python
from drangue import Agent, SQLiteStore

agent = Agent("claude-opus-4-8", tools=[book_flight], store=SQLiteStore("runs.db"))
result = await agent.run("Book my trip", run_id="trip-42")   # crash, rerun, same id -> resumes
```

A tool that causes a side effect can declare an `idempotency_key` parameter. The
runtime injects a stable key derived from the run and step (it never appears in
the model-facing schema), so the tool can deduplicate downstream:

```python
@tool
def book_flight(city: str, idempotency_key: str = "") -> str:
    """Book a flight."""
    return charge_once(city, key=idempotency_key)
```

## Cheap and local models

`drangue` ships two adapters. One of them, `OpenAIModel`, talks to any
OpenAI-compatible endpoint, which is most of the cheap and free backends. You
choose the backend with `base_url`; the agent loop does not change.

```python
from drangue import Agent, OpenAIModel

# Free and local. Install Ollama, run `ollama pull llama3.1`. No API key, no per-token cost.
agent = Agent(
    model=OpenAIModel("llama3.1", base_url="http://localhost:11434/v1", api_key="ollama"),
    tools=[get_weather],
)
```

Swap the model line for a cheap hosted backend without touching anything else:

| Backend | How |
| --- | --- |
| Ollama / LM Studio | `OpenAIModel("llama3.1", base_url="http://localhost:11434/v1", api_key="ollama")` (free, local) |
| DeepSeek | `OpenAIModel("deepseek-chat", base_url="https://api.deepseek.com")` |
| Groq | `OpenAIModel("llama-3.1-8b-instant", base_url="https://api.groq.com/openai/v1")` |
| OpenRouter | `OpenAIModel("...", base_url="https://openrouter.ai/api/v1")` |
| OpenAI | `OpenAIModel("gpt-4o-mini")` |
| Claude | `"claude-opus-4-8"` or `AnthropicModel("claude-opus-4-8")` |

`api_key` and `base_url` fall back to the `OPENAI_API_KEY` and
`OPENAI_BASE_URL` environment variables when omitted. See `examples/cheap.py`.

## Bring your own model

`model` can be a string (the default Anthropic adapter), one of the adapters
above, or any object with an async `generate` method. That seam is how you swap
providers, add caching, or pass a fake model in tests. The OpenAI and Anthropic
adapters are both tested fully offline against fake clients, see
`tests/test_openai_model.py` and the AnthropicModel tests in
`tests/test_cost.py`.

## What drangue does not do

Keeping the surface obvious is the point.

- No graph or DAG concept. You write a normal agent; the runtime drives the loop.
- No prompt-template engine. f-strings are fine.
- No built-in RAG or vector store. That is a different library.
- No provider-specific code in the core. Adapters are swappable.
- No mandatory config files.

## Architecture

The simple facade sits on a small, durable-by-design core: an **orchestrator**
decides each step deterministically, an **executor** performs it, and every step
is appended to a **store** as an event log. The log is the source of truth; the
run is a fold of it. That shape is what lets observability, durability, and
recovery layer on without changing the facade. See `ROADMAP.md`.

## Roadmap

The current focus is the production core (`ROADMAP.md`):

- Done: orchestrator/executor split, event log, async core.
- Done: observability (per-step timing and cost, a trace tree, console and
  OpenTelemetry tracers, reasoning capture).
- Done: durable resume after a crash (SQLite store, replay, idempotency keys,
  the three state scopes).
- Done: hardened tool calls (timeouts, retries with backoff, your own validation
  callback run before the model sees a field, clean structured failures,
  fallbacks).
- Done: cost and latency (per-run token and dollar budgets, model routing,
  prompt caching).
- Done: security and guardrails (permission scoping, action gates, input and
  output guards, reversibility metadata).
- Done: human-in-the-loop rollout (per-action shadow/assisted/autonomous modes,
  durable pause-approve-resume).
- Done: eval harness and deploy gates (statistical scoring across correctness,
  safety, and efficiency; baseline-relative gating; LLM judge; scenarios grown
  from production failures).
- Done: token streaming, multi-turn conversations, parallel sibling tool
  calls, structured output, recorded run config, per-run leases, and MCP
  servers as tool packs (drangue-mcp).

All of Chapters 4 to 12 are implemented (Phases 0 to 7).

## Develop

```bash
pip install -e ".[dev]"
python run_tests.py     # no pytest needed; uses a tiny async runner
```

Contributions are welcome under MIT with a DCO sign-off (`git commit -s`);
see `CONTRIBUTING.md`.

## License

MIT
