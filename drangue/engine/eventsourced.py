"""The default engine: drive the loop over an append-only event log.

The loop is the one the book describes. State is the result of replaying the
log; the process is disposable. In Phase 0 the store is in-memory, so "replay"
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
from ..orchestrator import Done, fold


class EventSourcedEngine:
    async def run(self, *, run_id, orchestrator, executor, store, system, input, emit=None) -> Result:
        if not await store.load(run_id):
            started = Event(seq=0, type="run_started", payload={"input": input})
            await store.append(run_id, started)
            if emit:
                await emit(started)

        while True:
            events = await store.load(run_id)
            state = fold(events)
            step = orchestrator.next(state)

            if isinstance(step, Done):
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
                    step, system=system, messages=state.messages, idempotency_key=key
                )
                await store.append(run_id, produced)
                if emit:
                    await emit(produced)
