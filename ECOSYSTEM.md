# drangue ecosystem

How drangue gets batteries without becoming a kitchen sink: a tiny core, and
batteries that plug into seams. The rule is **available, not included.**

Three invariants protect the wedge:

1. The core never depends on a battery.
2. Every battery plugs into a documented seam; it never patches the core.
3. The simple path (`Agent("model", tools=[...])`) never sees any of it.

If a proposed battery cannot satisfy all three, it does not go in the core, and
usually it does not go in the project at all.

## Tiers

- **Tier 0 - core (`drangue`).** The loop, the seams, and the zero-friction
  defaults (in-memory store, null tracer, single model, no guardrails). Near-zero
  dependencies. This tier defines the protocols; it is the constitution.
- **Tier 1 - first-party adapters.** Thin, seam-fitting implementations the core
  team maintains: model providers, durable stores, tracer exporters. Shipped as
  extras (`drangue[openai]`) or sibling packages. No core change, ever.
- **Tier 2 - first-party batteries.** Larger optional packages behind a seam:
  long-term memory, drift detection, a Temporal engine, an embedding judge. Own
  dependencies, own release cadence, same quality bar as core.
- **Tier 3 - community / contrib.** Domain tool packs (SRE tools, browser tools),
  guard libraries, check packs. Lower bar, clearly labeled experimental, must
  pass the conformance suite to be listed.
- **Tier 4 - platform (not a package).** A hosted dashboard over the traces, eval
  profiles, and cost the library already emits. Built *on top of* the library as
  a separate product, never inside it. Out of scope for this repo.

## The seams (the public contract)

Every battery implements one of these. The core ships a default for each, so a
battery is always strictly additive.

| Seam | Protocol (method) | Core default | Example batteries | Tier |
|---|---|---|---|---|
| **Model** | `async generate(...)` | `AnthropicModel`, `OpenAIModel` | Bedrock, Vertex, Gemini-native, Ollama-native | 1 |
| **Store** | `async append/load` | `InMemoryStore`, `SQLiteStore` | Postgres, Redis, DynamoDB | 1 |
| **Engine** | `async run(ctx)` | `EventSourcedEngine` | Temporal, DBOS, Restate adapters | 2 |
| **Tracer** | `span(name, /, **attrs)` | `NullTracer`, `ConsoleTracer`, `OTelTracer` | Langfuse, Phoenix, Braintrust exporters | 1-2 |
| **Memory** | `async recall/remember` | `NullMemory` | pgvector memory, Redis memory | 2 (needs core hook, below) |
| **Router** | `choose(messages, step_index)` | `SingleModel`, `RuleRouter` | cost-aware router, latency router | 2 |
| **Judge** | `async yes_no(question)` | `Judge` (wraps a Model) | embedding judge, rubric judge | 2 |
| **Tools** | `@tool` functions | (none) | drangue-mcp (any MCP server), SRE toolpack, web/browser tools | 3 |
| **Guard callables** | `input_guard` / `output_guard` / `approver` | (none) | injection scanner, PII redactor, Slack approver | 2-3 |
| **Eval checks** | `Check` builders | `output_contains`, `forbids_tool`, ... | domain check packs | 3 |

Note what is NOT a seam: `Budget`, `Guardrails`, and `Autonomy` are plain config
objects, not protocols. Batteries extend them by supplying callables (an
`approver`, an `input_guard`), not by subclassing.

## The seam contract

A battery MUST:

1. **Implement exactly one protocol** (Model, Store, Engine, Tracer, Memory,
   Router, Judge) OR contribute tools / checks / guard callables. One job.
2. **Match the protocol's async-ness.** If the seam method is `async`, the
   battery's is too.
3. **Require no core change.** If it needs one, that is a separate core PR first.
   A battery is never the reason the core grows.
4. **Be wired explicitly.** Construct it and pass it in. No import-time side
   effects, no global registry, no entry-point auto-discovery. Magic
   auto-wiring is the exact thing the wedge rejects.
5. **Be strictly additive.** The core default for the seam must already work, so
   installing the battery only adds capability, never changes the simple path.
6. **Own its dependencies.** The core depends on nothing in the battery; the
   battery pins a compatible core range (`drangue>=X,<Y`).
7. **Pass the conformance suite** for its seam (see below).
8. **Name itself by tier.** `drangue-<x>` for first-party, `<x>-drangue` or
   `drangue-contrib-<x>` for community.

## Wiring is explicit, always

```python
from drangue import Agent, Autonomy, Budget, Guardrails
from drangue_postgres import PostgresStore
from drangue_memory import PgVectorMemory
from drangue_langfuse import LangfuseTracer

agent = Agent(
    "claude-opus-4-8",
    tools=[...],
    store=PostgresStore(dsn),          # Store battery
    memory=PgVectorMemory(dsn),         # Memory battery
    tracer=LangfuseTracer(),            # Tracer battery
    budget=Budget(max_usd=0.50, prices=...),
    guardrails=Guardrails(input_guard=InjectionScanner(), approver=slack_approve),
    autonomy=Autonomy(modes={"deploy": "assisted"}),
)
```

No `pip install` silently changes behavior. You can read this constructor and
know exactly what is plugged in. That legibility is the product.

## The one core change the batteries need: DONE

`Memory` was a protocol the core never called. It is now wired into the loop,
gated behind `memory=None` so the simple path is byte-for-byte unchanged:

- **Recall is automatic and recorded.** If an Agent has a `memory`, the engine
  calls `recall(input)` before the first model step, records the result as a
  `memory_recalled` event, and `fold` injects it into the model's system context.
  Because it is a recorded fact, a resumed run replays it instead of re-querying,
  keeping the orchestrator a pure function of the log.
- **Writing is deliberate, not automatic.** `Agent.remember(item)` is an explicit
  call. The core does NOT auto-remember every run: "remember everything" is the
  anti-pattern Chapter 5 warns against, so the decision of what is worth keeping
  belongs to the application or the memory battery, not the loop.

That was the only core change needed to unlock Tier 2 memory. Everything else
(stores, tracers, routers, judges, tools, guards, drift) already has its seam.
Drift detection, notably, needs **no** core change: it is a pure consumer of the
persisted event log and traces the core already emits.

## Repository layout: a monorepo

First-party packages live in one repository, but each is a separate,
independently published package, not a submodule of core. Crucially,
`extensions/` is a sibling of the core package directory, not under it, so
nothing is importable as `drangue.extensions` and no extension's dependency ever
reaches core.

```
drangue/                         # repo root (monorepo)
  drangue/                       # Tier 0: core package (zero runtime deps)
  tests/                         # core tests
  pyproject.toml                 # core
  extensions/
    run_tests.py                 # runs every extension's tests in one place
    drangue-memory/              # Tier 2: Memory   -> import drangue_memory
      drangue_memory/  tests/  pyproject.toml
    drangue-postgres/            # Tier 1: Store    -> import drangue_postgres
    drangue-temporal/            # Tier 2: Engine
    drangue-mcp/                 # Tier 3: Tools    -> import drangue_mcp
```

Each extension has its own `pyproject.toml` pinning `drangue>=X,<Y` and is
meant to be published to PyPI on its own (`pip install drangue-postgres`);
release automation currently covers only the core, so extension releases are
manual until that lands. Keeping them in
one repo buys atomic cross-cutting changes (a battery that needs a seam change is
a single commit, as the RunSpec work was) and one CI, without giving up the
dependency boundary. Tier 3 community packages stay in their own repos,
discoverable via a list.

## Conformance tests: SHIPPED

The core publishes `drangue.testing`: a conformance suite per seam plus the
offline fakes (`FakeModel`, `RecordingTracer`). A battery author runs the check
for their seam and proves the contract holds. The checks are plain async
functions (no test-framework dependency), so they run under pytest, drangue's
own runner, or a script.

```python
from drangue.testing import check_store, check_store_idempotent_append

async def test_my_postgres_store_conforms():
    await check_store(lambda: PostgresStore(dsn))                 # required core
    await check_store_idempotent_append(lambda: PostgresStore(dsn))  # durable promise
    await check_store_with_agent(lambda: PostgresStore(dsn))      # end-to-end
```

Available checks: `check_store`, `check_store_idempotent_append`,
`check_store_with_agent`, `check_memory`, `check_tracer`, `check_router`,
`check_model_interface`. drangue's own built-ins are tested through these same
suites (`tests/test_conformance.py`), so the contract a battery is held to is the
exact contract the core meets. This is what lets Tier 3 stay trustworthy without
core-team review of every package.

## What stays out, on purpose, forever

- **A RAG / retrieval engine in core.** Retrieval is just tools plus memory.
  Bring LlamaIndex or your own; expose it as a `@tool`.
- **A prompt-template DSL.** f-strings are fine; templating is a different
  library.
- **A multi-agent orchestrator.** Chapter 13 is a deliberate non-goal. Compose
  `Agent`s yourself if a measured case proves separateness earns its cost.
- **A hosted platform inside the library.** That is Tier 4, a separate product.

## Distributed engines: the construction seam (resolved)

Building the first batteries tested the seams. Memory (`drangue-memory`) and
Store (`drangue-postgres`) fit cleanly, and the Engine seam accepts in-process
engines (`check_engine` passes for `EventSourcedEngine`).

Building `drangue-temporal` surfaced that a distributed runtime could not get a
`RunContext` on a remote worker: `Engine.run(ctx)` takes live, in-process objects
(the executor with its model client and tool callables), which cannot cross a
wire. `Engine.run(ctx)` itself was fine; the missing piece was a serializable way
to *reconstruct* the context on the worker.

That is now in core:

- **`RunSpec`** (serializable: `key`, `run_id`, `input`) is what crosses the
  boundary, with `to_dict` / `from_dict`.
- **`AgentRegistry`** lives on the worker and maps a key to a factory that
  rebuilds the live agent (model, tools, guardrails, memory) locally.
- **`context_from_spec(spec, registry)`** is the worker-side reconstruction: it
  builds a `RunContext` the same engine then drives.

So `Engine.run(ctx)` stays the single contract; a distributed engine just builds
its `ctx` from a `RunSpec` after the spec arrives. Proven offline by
`tests/test_distributed.py`, which JSON round-trips a spec, rebuilds on a
simulated worker, and resumes from a shared store without re-calling the model.

`extensions/drangue-temporal` is the real, **verified** engine built on this:
Temporal supervises the durable engine (a workflow drives the run, an activity
rebuilds the agent via the registry and runs `EventSourcedEngine`), so every
feature works unchanged and activity-retry composes with store-resume.
Human-in-the-loop becomes a durable wait on an approval signal. Both paths
(run-to-completion and pause-approve-resume) pass against Temporal's
time-skipping test server, and CI's `extensions` job runs that suite (with
`DRANGUE_TEMPORAL_TEST=1`) on every push. See `drangue-temporal/README.md`.

## Versioning and stability

The **seams are the public API.** They follow semver: changing a protocol's
signature is a major bump. The orchestrator and engine internals are not public
and can change in minor releases. Batteries declare the core range they support.
This is the contract that lets the ecosystem grow without the core having to.
