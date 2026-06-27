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

from .. import rollout
from ..events import Event, Result
from ..memory import item_to_dict, render_context
from ..observability import NullTracer
from ..orchestrator import Done, ModelStep, ToolStep, fold


def _compose_system(system: str, recalled: list) -> str:
    """Append recalled memory (a recorded fact) to the system context."""
    context = render_context(recalled)
    if not context:
        return system
    return f"{system}\n\n{context}" if system else context


class EventSourcedEngine:
    async def run(self, *, run_id, orchestrator, executor, store, system, input,
                  emit=None, tracer=None, budget=None, autonomy=None,
                  memory=None) -> Result:
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
                    events = await store.load(run_id)
                    output = f"(blocked: {reason})"
                    blocked = Event(seq=rollout.next_seq(events), type="run_finished",
                                    payload={"output": output, "blocked": True})
                    await store.append(run_id, blocked)
                    if emit:
                        await emit(blocked)
                    events = events + [blocked]
                    return Result(output=output, messages=fold(events).messages,
                                  events=events)

            while True:
                events = await store.load(run_id)
                state = fold(events)

                # Recall once, before the first model step. The result is
                # recorded, so a resumed run replays it instead of re-querying,
                # and the orchestrator stays a pure function of the log.
                if memory is not None and not state.recalled_done:
                    items = await memory.recall(input)
                    recalled = Event(seq=state.next_seq, type="memory_recalled",
                                     payload={"items": [item_to_dict(i) for i in items]})
                    await store.append(run_id, recalled)
                    if emit:
                        await emit(recalled)
                    continue

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

                # Per-action autonomy (Chapter 12): shadow proposes without
                # acting, assisted pauses for approval, autonomous falls through.
                if isinstance(step, ToolStep) and autonomy is not None:
                    action, paused = await self._apply_autonomy(
                        step, autonomy, events, state, run_id, store, run_span, emit)
                    if action == "paused":
                        return paused
                    if action == "handled":
                        continue
                    # action == "execute": fall through to normal execution.

                # Already recorded? Then this is a replay; fold it and move on.
                recorded = next((e for e in events if e.seq == step.seq), None)
                if recorded is None:
                    key = f"{run_id}:{step.seq}"
                    produced = await executor.run(
                        step, system=_compose_system(system, state.recalled),
                        messages=state.messages, idempotency_key=key, tracer=tracer,
                    )
                    await store.append(run_id, produced)
                    if emit:
                        await emit(produced)

    async def _apply_autonomy(self, step, autonomy, events, state, run_id, store,
                              run_span, emit):
        """Return (action, result): action is paused | handled | execute."""
        call = step.call
        mode = autonomy.mode_for(call.name)

        if mode == rollout.SHADOW:
            ev = Event(seq=step.seq, type="tool_result", payload={
                "call_id": call.id, "name": call.name,
                "content": rollout.shadow_result(call),
            })
            await store.append(run_id, ev)
            if emit:
                await emit(ev)
            return "handled", None

        if mode == rollout.ASSISTED:
            decision, reason = rollout.approval_decision(events, call.id)
            if decision == "denied":
                ev = Event(seq=step.seq, type="tool_result", payload={
                    "call_id": call.id, "name": call.name,
                    "content": rollout.denied_result(call, reason),
                })
                await store.append(run_id, ev)
                if emit:
                    await emit(ev)
                return "handled", None
            if decision == "pending":
                if not rollout.has_approval_request(events, call.id):
                    req = Event(seq=step.seq, type="approval_requested", payload={
                        "call_id": call.id, "tool": call.name,
                        "arguments": call.arguments, **rollout.last_reasoning(events),
                    })
                    await store.append(run_id, req)
                    if emit:
                        await emit(req)
                run_span.set("status", "paused")
                events = await store.load(run_id)
                return "paused", Result(output="(paused: awaiting approval)",
                                        messages=state.messages, events=events)
            # decision == "granted": execute the now-approved action.

        return "execute", None
