# drangue-temporal (unblocked: the construction seam landed)

Status: **implementable.** Building this battery found that a distributed runtime
could not reconstruct a run on a remote worker; the core now provides the
serializable construction path, so a Temporal engine can be built. The
implementation still needs a Temporal dev server to verify, so it is not shipped
here yet, but the blocker is gone.

## What was missing, and what landed

The Engine seam is `async run(ctx)`, where `ctx` is a `RunContext` carrying
**live, in-process objects** (the executor with its model client and tool
callables). Those cannot cross a wire, and Temporal needs serializable workflow
arguments with work running on pre-registered workers. `run(ctx)` itself was
fine; the gap was getting a `ctx` on the worker.

Core now provides the serializable construction path (in `drangue.context` and
`drangue.registry`):

- **`RunSpec`** (`key`, `run_id`, `input`) crosses the boundary.
- **`AgentRegistry`** on the worker maps the key to a factory that rebuilds the
  live agent locally.
- **`context_from_spec(spec, registry)`** reconstructs the `RunContext`.

The same `Engine.run(ctx)` then drives it. `tests/test_distributed.py` in core
proves the round-trip: a spec is JSON-serialized, rebuilt on a simulated worker,
run, and resumed from a shared store without re-calling the model.

## How the Temporal engine looks now

```
@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, spec: RunSpec) -> Result:
        # The orchestrator (pure) runs as deterministic workflow code.
        # Each executor step runs as an activity:
        while True:
            step = orchestrator.next(state)          # deterministic -> safe in a workflow
            if done: return result
            ev = await workflow.execute_activity(run_step, spec, step)  # side effects
            state = fold(state, ev)

@activity.defn
async def run_step(spec: RunSpec, step) -> Event:
    agent = registry[spec.key]()      # reconstruct model + tools on the worker
    return await agent.executor.run(step, ...)
```

Temporal's history is the durable log; our orchestrator's determinism rules (no
clock/random/IO) line up exactly with Temporal's workflow determinism rules. The
architecture is compatible and the construction seam now fits, so the only thing
left is the implementation against a live Temporal server.

## What this validates

The Store and Memory seams accept batteries cleanly (`drangue-postgres`,
`drangue-memory` both pass conformance). The Engine seam accepts in-process
engines (`check_engine` passes for `EventSourcedEngine`), and the new
RunSpec + registry path lets a distributed engine rebuild a run on a worker
(proven offline in `tests/test_distributed.py`). Building this battery is what
drove that core change: the seam grew because a real battery needed it.
