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

import time

from .. import rollout
from ..events import Event, Result
from ..memory import item_to_dict, live_items, render_context
from ..observability import NullTracer
from ..orchestrator import Done, ModelStep, ToolStep, fold


def _compose_system(system: str, recalled: list) -> str:
    """Append recalled memory (a recorded fact) to the system context."""
    context = render_context(recalled)
    if not context:
        return system
    return f"{system}\n\n{context}" if system else context


class EventSourcedEngine:
    async def run(self, ctx) -> Result:
        run_id = ctx.run_id
        orchestrator = ctx.orchestrator
        executor = ctx.executor
        store = ctx.store
        system = ctx.system
        input = ctx.input
        emit = ctx.emit
        budget = ctx.budget
        autonomy = ctx.autonomy
        memory = ctx.memory
        tracer = ctx.tracer or NullTracer()

        with tracer.span("run", run_id=run_id) as run_span:
            existing = await store.load(run_id)
            if not existing:
                started = Event(seq=0, type="run_started", payload={"input": input})
                await store.append(run_id, started)
                if emit:
                    await emit(started)
            elif input:
                recorded = next((e.payload.get("input", "") for e in existing
                                 if e.type == "run_started"), "")
                if input != recorded:
                    # Silently replaying the old run's answer for a new
                    # question is the worst possible interpretation; refuse.
                    raise ValueError(
                        f"run {run_id!r} already exists with a different input "
                        f"({recorded!r}). Follow-up messages into an existing "
                        "run are not supported; resume it with "
                        "agent.resume(run_id) or start a new run_id."
                    )

            # Input guardrail: a blocked input ends the run before any model
            # work. This runs from the RECORDED input, so a resumed run that
            # crashed before its first step is vetted too, not just fresh ones.
            events = await store.load(run_id)
            state = fold(events)
            if state.model_calls == 0 and not state.finished:
                reason = await executor.check_input(state.input)
                if reason is not None:
                    output = f"(blocked: {reason})"
                    blocked = Event(seq=state.next_seq, type="run_finished",
                                    payload={"output": output, "blocked": True})
                    await store.append(run_id, blocked)
                    if emit:
                        await emit(blocked)
                    events = events + [blocked]
                    return Result(output=output, messages=state.messages,
                                  events=events)

            while True:
                events = await store.load(run_id)
                state = fold(events)

                # Recall once, before the first model step. The result is
                # recorded, so a resumed run replays it instead of re-querying,
                # and the orchestrator stays a pure function of the log. Expiry is
                # honored here against a recorded timestamp; a failing recall is
                # non-fatal (record empty and proceed without memory). Recall
                # keys off the RECORDED input, so a resumed run queries the
                # same memory the original would have.
                if memory is not None and not state.recalled_done:
                    now = time.time()
                    error = None
                    try:
                        items = await memory.recall(state.input)
                    except Exception as exc:  # an enhancement must never crash the run
                        items, error = [], str(exc)
                    payload = {
                        "items": [item_to_dict(i) for i in live_items(items, now)],
                        "recalled_at": now,
                    }
                    if error is not None:
                        payload["error"] = error
                    recalled = Event(seq=state.next_seq, type="memory_recalled",
                                     payload=payload)
                    await store.append(run_id, recalled)
                    if emit:
                        await emit(recalled)
                    continue

                step = orchestrator.next(state)

                # Enforce the budget before an expensive (model) step, using the
                # spend already recorded in the log. Finish gracefully instead of
                # starting work the run cannot afford.
                if isinstance(step, ModelStep) and budget is not None:
                    try:
                        over = budget.exceeded(events)
                    except ValueError as exc:
                        # Spend cannot be computed (an unpriced model in the
                        # log). Fail closed, but as a recorded graceful stop
                        # before more spend — not a crash that leaves the run
                        # looking alive.
                        step = Done(f"(stopped: budget unenforceable: {exc})")
                    else:
                        if over:
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
