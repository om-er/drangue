# drangue production roadmap

Bringing drangue in line with the principles in *Production AI Agents*, without
losing the "tiny, obvious" wedge. This document covers the **Production Core**:
Phases 0 to 3. Later phases (cost, security, human-in-the-loop, evals) are
sketched at the end but are out of scope for this round.

## The organizing principle

The book argues an agent must be a system, not a thin wrapper. drangue's value
is that it feels like a thin wrapper. We reconcile the two with one rule:

> Keep the simple facade as the default. Re-implement it on top of a durable
> orchestrator and executor. Every production property is additive and opt-in.

The simple path is just the production path configured with an in-memory store,
no tracer, and auto-approved actions.

```python
# Default. The wedge, preserved. Unchanged for users.
agent = Agent("claude-opus-4-8", tools=[get_weather])
await agent.run("What should I wear in Paris?")

# Production. Same object, opt-in seams.
agent = Agent(
    "claude-opus-4-8",
    tools=[get_weather],
    store=SQLiteStore("runs.db"),     # Ch4/5 durability + state
    tracer=OTelTracer(),               # Ch9 observability
    engine=EventSourcedEngine(),       # default; swap for TemporalEngine(...)
)
await agent.run("...", run_id="ticket-4821")   # resumes after a crash
```

## Decisions for this round

- **Concurrency: async core.** `run` and `stream` become coroutines. A thin
  `run_sync` convenience preserves the 3-line script experience.
- **Durability: build our own, pluggable.** Default is a minimal event-sourced
  engine over a `Store`. The `Engine` protocol is the seam: an adapter can hand
  orchestration to Temporal or DBOS without touching agent code.
- **Scope: Phases 0 to 3.** This delivers the backbone of all five
  non-negotiable properties (Ch1): correctness substrate, recoverability,
  observability, and the start of cost-predictability and governability.

## Target module layout

```
drangue/
  __init__.py
  agent.py            # facade: wires engine + executor, exposes run/stream/run_sync
  tool.py             # @tool, now async-aware (sync tools run in a thread)
  models.py           # async Model ABC, AnthropicModel, OpenAIModel
  events.py           # Event/Step schema: the durable log's records
  orchestrator.py     # pure, deterministic: decide next step from the log
  executor.py         # side-effecting: run a step, apply idempotency
  engine/
    __init__.py       # Engine protocol (the pluggable seam)
    eventsourced.py   # default EventSourcedEngine (append-before-act, replay)
  store/
    __init__.py       # Store protocol
    memory.py         # InMemoryStore (default)
    sqlite.py         # SQLiteStore (durable)
  observability/
    __init__.py       # Tracer protocol + span model
    console.py        # ConsoleTracer (replaces today's trace=True)
    otel.py           # optional OpenTelemetry exporter
  tools/
    wrap.py           # defensive wrapper: timeout, retry, validate, clean failure
```

## Core interfaces (async)

```python
# events.py - the log is the source of truth (Ch4, Ch5)
@dataclass
class Event:
    seq: int                 # position in the run's log; basis for idempotency
    type: str                # run_started | model_decision | tool_result | run_finished | run_failed
    payload: dict
    ts: str                  # recorded as a fact by the executor, never read during replay

# orchestrator.py - pure and deterministic (Ch4)
class Orchestrator:
    def next(self, state: State) -> Step | Done:
        # Pure function of the replayed log. No clock, no random, no I/O.
        ...

# executor.py - side-effecting and non-deterministic (Ch4)
class Executor:
    async def run(self, step: Step, *, idempotency_key: str) -> Event:
        # Model calls and tool calls happen here, behind timeouts and retries.
        ...

# engine/__init__.py - the pluggable durability seam (default 1, allow 2)
class Engine(Protocol):
    async def run(self, *, run_id, orchestrator, executor, store, tracer) -> Result: ...

# store/__init__.py - where the log lives
class Store(Protocol):
    async def append(self, run_id: str, event: Event) -> None: ...
    async def load(self, run_id: str) -> list[Event]: ...

# observability/__init__.py
class Tracer(Protocol):
    def span(self, name: str, **attrs): ...   # context manager; no-op by default
```

## The loop, as the book describes it

The default `EventSourcedEngine.run` drives a state machine. State is the result
of replaying the log; the process is disposable.

```
events = await store.load(run_id)          # replay to reconstruct state
state  = fold(events)
while True:
    step = orchestrator.next(state)        # deterministic decision
    if step is Done:
        return Result.from_(state)
    recorded = state.result_for(step.seq)  # already in the log?
    if recorded is None:                   # first run: consult the oracle
        key      = f"{run_id}:{step.seq}"  # stable idempotency key
        recorded = await executor.run(step, idempotency_key=key)
        await store.append(run_id, recorded)   # append before continuing
    state = fold(state, recorded)          # every replay reads the recorded fact
```

The model is non-deterministic the first time and a recorded fact every replay
after. That is how a non-deterministic agent runs on infrastructure that
requires determinism.

---

## Phase 0: The split, the log, and async  (done)

**Goal.** Re-architect today's fused loop into orchestrator plus executor with an
event log, and move the whole stack to async. No new production behavior yet;
the in-memory log is not persisted across processes. (Ch2, Ch4)

**Changes.**
- Make `Model.generate`, the tool dispatch, and `Agent.run`/`stream` async.
- Async adapters: `AsyncAnthropic`, `AsyncOpenAI`. Sync tools run via
  `asyncio.to_thread`; async tools are awaited.
- Add `events.py`, `Orchestrator.next`, `Executor.run`, and an in-process
  `EventSourcedEngine` backed by `InMemoryStore`.
- Add `run_sync()` so simple scripts stay one line.

**New/changed API.** `await agent.run(...)`, `async for e in agent.stream(...)`,
`agent.run_sync(...)`.

**Tests.** Port the existing offline tests to async; assert the event log
contents (the right sequence of `model_decision` and `tool_result` events).

**Done when.** Behavior matches today, but the architecture is the split with a
real event log, fully async, and the public facade is unchanged in spirit.

---

## Phase 1: Observability  (done)

**Goal.** Make a run fully reconstructable from the outside, with cost and
timing. This is the substrate the rest of the book stands on. (Ch9)

**Changes.**
- Capture token usage and latency on every model and tool step; add `usage` to
  `ModelResponse` and totals to `Result`.
- `Tracer` protocol plus `ConsoleTracer` (replaces the `trace=True` flag) and an
  optional `OTelTracer` building OpenTelemetry-shaped spans:
  run span -> step spans -> model/tool spans.
- Semantic logging hook: the orchestrator can attach a short reasoning string to
  each `model_decision` event (intent, not just mechanics).
- Persist the trace alongside the log so failures can be reopened later.

**Tests.** A run yields a complete trace: full prompts, results, token counts,
per-step latency, and a navigable span tree.

**Done when.** You can reconstruct any run end to end from the recorded trace
without reading source, and `Result` reports total tokens and cost.

---

## Phase 2: Durability and recovery  (done)

**Goal.** A run survives process death and resumes exactly where it stopped, with
no duplicated side effects. The book's central chapter. (Ch4, Ch5)

**Changes.**
- `SQLiteStore`: append-only durable log.
- `EventSourcedEngine`: append-before-act and replay-to-resume by `run_id`;
  recorded model decisions are replayed as facts, never re-asked.
- Determinism discipline: document the rule (no clock, random, or I/O in
  `Orchestrator.next`) and add a replay test that asserts a resumed run makes the
  same decisions as the original.
- Idempotency keys `f(run_id, seq)` threaded into the executor and passed to
  tools; retries reuse the key.
- State scopes (Ch5): task state is the durable log; conversation state is a
  derived view regenerated from it; long-term memory is an optional store with a
  stubbed interface and an explicit staleness contract.

**Tests.** Stop a run after N events, start a fresh process, resume, and assert:
identical result, the model is not re-called for recorded steps, and a
side-effecting tool runs exactly once across the crash (idempotency).

**Done when.** A crash-resume test passes and recovered runs are provably
coherent.

---

## Phase 3: Tool hardening  (done)

**Goal.** Every integration is bounded and returns structured outcomes, never raw
exceptions or plausible garbage. (Ch6)

**Changes.**
- A defensive wrapper applied by the executor, opt-in per tool via `@tool(...)`
  options or a `wrap(...)` helper:
  1. timeout tuned to the operation,
  2. classify failures (transient, permanent, auth),
  3. retry transient with backoff, reusing the idempotency key,
  4. validate the response against a schema before any field is read,
  5. return clean failure, fallback, or partial result.
- Failures come back to the model as structured data it can reason about, which
  the current `_dispatch` already does in spirit; this makes it rigorous.

**Tests.** Inject timeout, HTTP 429, malformed body, and schema drift into a fake
backend; assert the wrapper retries, backs off, validates, and returns clean
failures, never garbage.

**Done when.** A tool can hit every Chapter 6 failure mode and the agent always
receives a clean, categorized outcome.

---

## Milestone check against Chapter 1

After Phases 0 to 3, drangue will have:

- **Correct under load:** the trace substrate that eval capture needs (full
  delivery of correctness comes with the eval phase).
- **Recoverable:** yes, event-sourced resume with idempotency.
- **Observable:** yes, complete persisted traces with cost and timing.
- **Cost-predictable:** measurement in place (token and latency capture);
  enforcement (budgets) lands in Phase 4.
- **Governable:** the executor seam where gates and scoping attach exists;
  the gates themselves land in Phases 5 and 6.

## Out of scope this round (later phases)

- **Phase 4, Cost and latency (Ch10):** DONE. Per-run token and dollar budgets
  enforced before expensive steps (Budget), model routing (Router / RuleRouter),
  model recorded per step, and prompt caching of the stable prefix.
- **Phase 5, Security and guardrails (Ch11):** DONE. Permission scoping
  (allow/deny) and action gates enforced in the executor, input and output
  guards, fail-closed approval, and reversibility metadata on tools (Guardrails).
- **Phase 6, Human-in-the-loop and rollout (Ch12):** per-action autonomy modes,
  pause-persist-resume approval built on Phase 2, approvals recorded as signals.
- **Phase 7, Evals and gates (Ch7, Ch8):** step and trajectory evals, validated
  LLM-as-judge, traced failures captured as scenarios, deploy gates with variance
  thresholds.
- **Multi-agent (Ch13):** a deliberate non-goal. Default to tool composition;
  only revisit if a measured case proves separateness is the value.

Phases 1 and 2 are intentionally the substrates these later phases require: the
tracer feeds evals and drift detection, and the durable log is what approval
pauses and resumes on.

## Risks and mitigations

- **Async migration changes the public signature (await).** Mitigate with
  `run_sync` so the simplest scripts stay simple.
- **Determinism is easy to violate.** Mitigate with a documented rule and a
  replay-divergence test in Phase 2.
- **Feature creep erodes the wedge.** Mitigate by wiring every default
  (in-memory store, no-op tracer, auto-approve) so the 3-line agent never has to
  know the production seams exist.
