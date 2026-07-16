# drangue production roadmap

Bringing drangue in line with the principles in *Production AI Agents*, without
losing the "tiny, obvious" wedge. Phases 0 to 3 were the **Production Core**;
the later phases (cost, security, human-in-the-loop, evals) have since landed
too. Phase status is recorded per section below.

This is a record of intent and sequencing, not an API reference. Where it
describes a shape, trust the code over this document.

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
    engine=EventSourcedEngine(),       # default; the Engine seam is pluggable
)
await agent.run("...", run_id="ticket-4821")   # resumes after a crash
```

Handing orchestration to external infrastructure happens through the same
`engine=` seam, but not always as a drop-in object: the Temporal adapter
supervises the durable engine from a workflow rather than replacing it. See
`extensions/drangue-temporal/README.md` for the shape it actually takes.

## Decisions for this round

- **Concurrency: async core.** `run` and `stream` become coroutines. A thin
  `run_sync` convenience preserves the 3-line script experience.
- **Durability: build our own, pluggable.** Default is a minimal event-sourced
  engine over a `Store`. The `Engine` protocol is the seam: an adapter can hand
  orchestration to Temporal or DBOS without touching agent code.
- **Scope: Phases 0 to 3.** This delivers the backbone of all five
  non-negotiable properties (Ch1): correctness substrate, recoverability,
  observability, and the start of cost-predictability and governability.

## The shape: who does what

This roadmap describes intent, not signatures. **The code is the source of truth
for every interface**; the sections below say what each piece is *for* and why it
exists, and point at where to read it. For the seam-by-seam public contract
(protocol, core default, example batteries), see `ECOSYSTEM.md`.

| Piece | Job | Read |
|---|---|---|
| **Agent** | The facade. Wires engine + executor; exposes `run` / `stream` / `run_sync`. | `drangue/agent.py` |
| **Event** | The durable log's record. The log is the source of truth (Ch4, Ch5). | `drangue/events.py` |
| **Orchestrator** | Pure and deterministic: decide the next step from the replayed log. No clock, no random, no I/O. | `drangue/orchestrator.py` |
| **Executor** | Side-effecting and non-deterministic: perform a step behind timeouts and retries, apply idempotency. | `drangue/executor.py` |
| **Engine** | The pluggable durability seam. Takes a `RunContext` so the seam stays stable as capabilities are added. | `drangue/engine/` |
| **Store** | Where the log lives: append and load. | `drangue/store/` |
| **Tracer** | Spans, no-op by default. | `drangue/observability/` |
| **Tool hardening** | Defensive wrapper: timeout, classify, retry, validate, clean failure. Opt in with `harden(...)`. | `drangue/hardening.py`, `drangue/tool.py` |

Modules added by later phases, each behind a default so the simple path never
sees them: `budget.py` (Ch10 spend limits), `routing.py` (Ch10 model routing),
`guardrails.py` (Ch11 scoping and guards), `rollout.py` (Ch12 autonomy and
approvals), `evals.py` (Ch7/8 scoring and deploy gates), `memory.py` (Ch5
long-term memory), `errors.py` (Ch6 failure categories), `context.py`
(`RunContext` / `RunSpec`), `registry.py` (worker-side construction for
distributed engines), and `testing/` (conformance suites and offline fakes).

Out-of-tree batteries live in `extensions/`: `drangue-postgres` (Store),
`drangue-memory` (Memory), `drangue-temporal` (Engine). Per `ECOSYSTEM.md`, the
core never depends on them.

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

That sketch is the load-bearing idea, not the signature. The real loop in
`drangue/engine/eventsourced.py` has the same spine and also threads budget,
autonomy, memory, and guardrails through it.

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
- Capture latency on every model and tool step, and token usage on every model
  step (tools spend no tokens); add `usage` to `ModelResponse` and totals to
  `Result`.
- `Tracer` protocol plus `ConsoleTracer` (replaces the `trace=True` flag) and an
  optional `OTelTracer` building OpenTelemetry-shaped spans: a run span with a
  model/tool span per step under it.
- Semantic logging: a `model_decision` carries a short `reasoning` string
  (intent, not just mechanics), which the approval surface reads back. This is
  passthrough from the model adapter, not something the orchestrator authors —
  it is populated from Anthropic thinking blocks and is `None` on models that
  expose no equivalent, so treat it as a bonus, not a guarantee.
- Keep the trace reconstructable from the log rather than persisting spans:
  `Result.trace` rebuilds the tree from recorded facts, so a failure can be
  reopened later with no tracer configured.

**Tests.** A run yields a complete trace: full prompts, results, token counts,
per-step latency, and a navigable span tree.

**Done when.** You can reconstruct any run end to end from the recorded trace
without reading source, `Result` reports total tokens, and `Result.cost(prices)`
reports spend (prices are a caller input: they vary by provider and drift, so
the library ships no table).

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
  derived view regenerated from it; long-term memory is an optional store behind
  the `Memory` seam with an explicit staleness contract (`NullMemory` by default;
  recall is recorded to the log so replay reads it as a fact).

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
  options or the `harden(...)` helper:
  1. timeout tuned to the operation,
  2. classify failures (transient, permanent, auth),
  3. retry transient with backoff, reusing the idempotency key,
  4. validate the response before any field is read,
  5. return a clean failure or a fallback.
- Validation is a callback (`validate=`), not a schema language. A declarative
  output schema would mean a validation dependency in the core, which the tier-0
  rule forbids and the "no Pydantic required" promise rules out. You write the
  check; the wrapper guarantees it runs before the model sees a field, and turns
  a raised `ValidationError` into a categorized failure like any other.
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

## Later phases (all landed since)

- **Phase 4, Cost and latency (Ch10):** DONE. Per-run token and dollar budgets
  enforced before expensive steps (Budget), model routing (Router / RuleRouter),
  model recorded per step, and prompt caching of the stable prefix
  (`Agent(..., cache=True)`; Anthropic-only, since OpenAI caches server-side
  with no client-side markers to set). A dollar budget is refused unless it can
  actually be computed — see the price-table rule in `budget.py`.
- **Phase 5, Security and guardrails (Ch11):** DONE. Permission scoping
  (allow/deny) and action gates enforced in the executor, input and output
  guards, fail-closed approval, and reversibility metadata on tools (Guardrails).
- **Phase 6, Human-in-the-loop and rollout (Ch12):** DONE. Per-action autonomy
  (shadow/assisted/autonomous), durable pause-approve-resume built on the event
  log, approval surface that carries the agent's reasoning (Autonomy).
- **Phase 7, Evals and gates (Ch7, Ch8):** DONE. Statistical multi-run scoring
  across correctness/safety/efficiency — rates carry the standard error that
  says how solid they are, and the Gate warns when its own threshold sits
  inside the noise band. Rule checks plus a narrow LLM judge,
  scenario_from_result to grow the set from production failures, and a
  baseline-relative deploy Gate with per-dimension thresholds and overrides that
  must state a reason. Persisting an override is the caller's job through the
  `record` seam; the Gate runs at deploy time and owns no store.
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

## Known limitations and follow-ups

From code review. None are blockers; they are the gap between "green in tests"
and the production-grade durability the docstrings advertise.

- **Serial sibling tool calls.** When a model requests several tools in one
  decision, the orchestrator returns them one ToolStep at a time, so they run
  sequentially. Parallelizing siblings is the obvious async win left, but it
  interacts with per-tool guardrails and assisted-mode pauses (some siblings may
  need approval while others run), so it needs care. Deterministic seq assignment
  by tool-call order makes it tractable. Not yet done.
- **Model-call exactly-once: partial, backend-dependent.** A stable key
  (`f"{run_id}:{step.seq}"`) is threaded into `model.generate`. OpenAI-compatible
  backends honor it as an `Idempotency-Key`, so a crash between the call returning
  and the decision being appended is exactly-once on resume (within the provider's
  window). Anthropic's Messages API has NO request idempotency key, so the default
  model is AT-LEAST-ONCE there: resume may re-call and double-charge. Mitigation:
  keep downstream effects idempotent (tools already do, via wants_idempotency_key),
  or add a "model-call started" marker before the call so resume can detect an
  in-flight call. The orchestrator-side replay still reuses a recorded decision;
  this gap is only the crash window before the decision is recorded.
- **Sync-tool timeouts bound the caller, not the work.** `wait_for` cannot cancel
  a worker thread, so a timed-out sync tool keeps running and a retry can run
  concurrently with it. Documented in `hardening.py`. Truly bounded execution
  needs a cancellable async tool or an out-of-process worker.
- **Budgets are a soft ceiling.** Enforcement reads recorded usage, so the step
  that crosses the limit still runs; the next one is refused. Overshoot is bounded
  by one model call. Documented in `budget.py`.
- **`reasoning` is provider-dependent.** A `model_decision` carries the model's
  stated intent, and the assisted-mode approval surface shows it. It is
  populated from Anthropic thinking blocks; the OpenAI Chat Completions API has
  no equivalent, so on that adapter the approver falls back to the decision's
  text. Nothing in the core authors a reasoning string, and deriving one from
  the output would be a summary posing as intent. Documented in `models.py`.
- **No partial-result contract.** A hardened tool returns a clean failure or a
  fallback. There is no first-class "partial" outcome; a `validate` callback can
  coerce a value, but nothing in the wrapper models partial success as distinct
  from success. Fine so far because tools are single-shot, but a paging or
  batch tool would want it.
- **Overrides are recordable, not recorded.** `GateDecision.override` demands a
  reason and hands a full audit record to a `record` sink, but the core ships no
  sink: pass one or the trail ends with the returned object. The Gate is
  deploy-time and has no store, so wiring it to one is deliberately the caller's
  choice.
