"""The default engine: drive the loop over an append-only event log.

The loop is the one the book describes. State is the result of replaying the
log; the process is disposable. In Phase 0/1 the store is in-memory, so "replay"
happens within a process. In Phase 2 the same loop, with a durable store,
survives process death and resumes exactly.

    replay log -> reconstruct state
    while not done:
        step = orchestrator.next(state)        # deterministic
        if step already recorded: fold and continue   # replay path
        else: execute, append before continuing        # first-run path
"""

from __future__ import annotations

from ..events import Event, Result
from ..observability import NullTracer
from ..orchestrator import Done, ModelStep, fold


class EventSourcedEngine:
    async def run(self, *, run_id, orchestrator, executor, store, system, input,
                  emit=None, tracer=None, budget=None) -> Result:
        tracer = tracer or NullTracer()

        with tracer.span("run", run_id=run_id) as run_span:
            if not await store.load(run_id):
                started = Event(seq=0, type="run_started", payload={"input": input})
                await store.append(run_id, started)
                if emit:
                    await emit(started)

                # Input guardrail: a blocked input ends the run before any work.
                reason = await executor.check_input(input)
                if reason is not None:
                    blocked = Event(seq=1, type="run_finished",
                                    payload={"output": f"(blocked: {reason})", "blocked": True})
                    await store.append(run_id, blocked)
                    if emit:
                        await emit(blocked)

            while True:
                events = await store.load(run_id)
                state = fold(events)
                step = orchestrator.next(state)

                # Enforce the budget before an expensive (model) step, using the
                # spend already recorded in the log. Finish gracefully instead of
                # starting work the run cannot afford.
                if (isinstance(step, ModelStep) and budget is not None
                        and budget.exceeded(events)):
                    step = Done("(stopped: budget exhausted)")

                if isinstance(step, Done):
                    run_span.set("output", step.output)
                    if not any(e.type == "run_finished" for e in events):
                        finished = Event(seq=state.next_seq, type="run_finished",
                                         payload={"output": step.output})
                        await store.append(run_id, finished)
                        if emit:
                            await emit(finished)
                        events = await store.load(run_id)
                    return Result(output=step.output, messages=state.messages, events=events)

                # Already recorded? Then this is a replay; fold it and move on.
                recorded = next((e for e in events if e.seq == step.seq), None)
                if recorded is None:
                    key = f"{run_id}:{step.seq}"
                    produced = await executor.run(
                        step, system=system, messages=state.messages,
                        idempotency_key=key, tracer=tracer,
                    )
                    await store.append(run_id, produced)
                    if emit:
                        await emit(produced)
