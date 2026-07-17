"""The default engine: drive the loop over an append-only event log.

The loop is the one the book describes. State is the result of replaying the
log; the process is disposable. In Phase 0/1 the store is in-memory, so "replay"
happens within a process. In Phase 2 the same loop, with a durable store,
survives process death and resumes exactly.

    replay log -> reconstruct state
    while not done:
        step = orchestrator.next(state)        # deterministic; replay is just
        execute, append, fold the new event    # folding recorded events first

Two windows get special treatment:

  - A consumer breaking out of `agent.stream` cancels the engine task. If the
    cancellation lands between a completed side effect and its append, the log
    would say the step never happened and a resume would re-execute it — so
    the append of an executed step is shielded from cancellation and always
    completes.
  - A crash between executing and recording cannot be shielded. For tools
    marked `reversible=False`, a `step_started` intent event is appended
    BEFORE execution; a resume that finds the intent without a result knows
    the outcome is unknown and refuses to blindly re-execute (the model gets
    a clean `unknown_outcome` failure instead). Reversible tools and model
    calls stay at-least-once, as documented.

What this does NOT provide: a lease. Two live processes driving the same
run_id concurrently are serialized per-event by the store's ConflictError
(the loser discards its result and folds the winner's), but both may still
execute a step before one loses the append race.
"""

from __future__ import annotations

import asyncio
import time

from .. import rollout
from ..errors import ConflictError
from ..events import Event, Result
from ..hardening import _classify, unknown_outcome_error
from ..memory import item_to_dict, live_items, render_context
from ..observability import NullTracer
from ..orchestrator import Done, ModelStep, ToolStep, fold


def _compose_system(system: str, recalled: list) -> str:
    """Append recalled memory (a recorded fact) to the system context."""
    context = render_context(recalled)
    if not context:
        return system
    return f"{system}\n\n{context}" if system else context


async def _append(store, run_id: str, event: Event, emit) -> bool:
    """Append an event; False when another writer already claimed the seq.

    A False return means our result LOST a write race (a concurrent resume, or
    an approval landing mid-step). The store kept the winner; the caller's move
    is to discard its own result, reload, and re-decide from the winner's log —
    never to overwrite or crash.
    """
    try:
        await store.append(run_id, event)
    except ConflictError:
        return False
    if emit:
        await emit(event)
    return True


async def _append_shielded(store, run_id: str, event: Event, emit) -> bool:
    """Like _append, but the write completes even if this task is cancelled.

    Used for the record of an EXECUTED step: once the side effect has
    happened, losing its append (a consumer breaking out of `stream` cancels
    the engine mid-await) would make a resume re-execute it.
    """
    write = asyncio.ensure_future(store.append(run_id, event))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # We are being cancelled; the shielded write is not. Let it land
        # before unwinding, then propagate the cancellation.
        if not write.done():
            try:
                await write
            except Exception:  # noqa: BLE001 - already unwinding on cancel
                pass
        raise
    except ConflictError:
        return False
    if emit:
        await emit(event)
    return True


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
            events = await store.load(run_id)
            if not events:
                started = Event(seq=0, type="run_started", payload={"input": input})
                await store.append(run_id, started)
                if emit:
                    await emit(started)
                events = [started]
            elif input:
                recorded = next((e.payload.get("input", "") for e in events
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

            # Intent events THIS process appended. An unresolved step_started
            # that is NOT in here came from a dead process, and its outcome is
            # unknown (see the irreversible-tool handling below).
            started_here: set = set()

            while True:
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
                    if await _append(store, run_id, recalled, emit):
                        events = events + [recalled]
                    else:
                        events = await store.load(run_id)   # fold the winner's
                    continue

                step = orchestrator.next(state)

                # Input guardrail: vet the RECORDED input once, right before
                # the first model step. Keying off the log (not the fresh-run
                # branch) means a resumed run that crashed before its first
                # step is vetted too.
                if isinstance(step, ModelStep) and state.model_calls == 0:
                    reason = await executor.check_input(state.input)
                    if reason is not None:
                        output = f"(blocked: {reason})"
                        blocked = Event(seq=state.next_seq, type="run_finished",
                                        payload={"output": output, "blocked": True})
                        if not await _append(store, run_id, blocked, emit):
                            events = await store.load(run_id)
                            continue
                        run_span.set("output", output)
                        return Result(output=output, messages=state.messages,
                                      events=events + [blocked])

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
                        if not await _append(store, run_id, finished, emit):
                            events = await store.load(run_id)
                            continue   # a concurrent writer got there; re-decide
                        events = events + [finished]
                    return Result(output=step.output, messages=state.messages, events=events)

                # Per-action autonomy (Chapter 12): shadow proposes without
                # acting, assisted pauses for approval, autonomous falls through.
                if isinstance(step, ToolStep) and autonomy is not None:
                    action, paused = await self._apply_autonomy(
                        step, autonomy, events, state, run_id, store, run_span, emit)
                    if action == "paused":
                        return paused
                    if action == "handled":
                        events = await store.load(run_id)
                        continue
                    # action == "execute": fall through to normal execution.

                # Irreversible tools get an intent marker BEFORE execution, so
                # a crash between the side effect and its record is detectable.
                if isinstance(step, ToolStep):
                    tool = executor.tools.get(step.call.name)
                    if tool is not None and not tool.reversible:
                        if (step.call.id in state.started
                                and step.call.id not in started_here):
                            # Orphaned intent from a dead process: the effect
                            # may or may not have happened. Refuse to blindly
                            # re-execute; give the model a clean failure.
                            ev = Event(seq=state.next_seq, type="tool_result",
                                       payload={
                                           "call_id": step.call.id,
                                           "name": step.call.name,
                                           "content": unknown_outcome_error(step.call.name),
                                       })
                            if await _append(store, run_id, ev, emit):
                                events = events + [ev]
                            else:
                                events = await store.load(run_id)
                            continue
                        if step.call.id not in state.started:
                            intent = Event(seq=state.next_seq, type="step_started",
                                           payload={"call_id": step.call.id,
                                                    "tool": step.call.name})
                            if await _append(store, run_id, intent, emit):
                                events = events + [intent]
                                started_here.add(step.call.id)
                            else:
                                events = await store.load(run_id)
                            continue
                        # else: intent recorded by us this process — execute.

                key = f"{run_id}:{step.seq}"
                try:
                    produced = await executor.run(
                        step, system=_compose_system(system, state.recalled),
                        messages=state.messages, idempotency_key=key, tracer=tracer,
                    )
                except Exception as exc:
                    # Record the failure before propagating it, so the
                    # persisted run reads "failed", not "running" forever.
                    # The exception still reaches the caller; a later
                    # resume retries from this point (fold skips the
                    # run_failed marker).
                    failed = Event(seq=state.next_seq, type="run_failed",
                                   payload={
                                       "error": str(exc) or type(exc).__name__,
                                       "category": _classify(exc),
                                   })
                    # Best-effort: if a concurrent writer claimed the seq,
                    # the log already moved on; the original exception is
                    # what matters.
                    await _append(store, run_id, failed, emit)
                    raise
                # The step has EXECUTED; its record must land even if we are
                # cancelled right now (a consumer breaking out of stream()).
                if not await _append_shielded(store, run_id, produced, emit):
                    # Our step lost a write race (a concurrent resume of this
                    # run_id). The winner's event is truth; discard ours,
                    # reload, and continue from the recorded log.
                    events = await store.load(run_id)
                    continue
                events = events + [produced]

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
            await _append(store, run_id, ev, emit)
            return "handled", None   # conflict or not: reload and re-decide

        if mode == rollout.ASSISTED:
            decision, reason = rollout.approval_decision(events, call.id)
            if decision == "denied":
                ev = Event(seq=step.seq, type="tool_result", payload={
                    "call_id": call.id, "name": call.name,
                    "content": rollout.denied_result(call, reason),
                })
                await _append(store, run_id, ev, emit)
                return "handled", None
            if decision == "pending":
                if not rollout.has_approval_request(events, call.id):
                    req = Event(seq=step.seq, type="approval_requested", payload={
                        "call_id": call.id, "tool": call.name,
                        "arguments": call.arguments, **rollout.last_reasoning(events),
                    })
                    if not await _append(store, run_id, req, emit):
                        return "handled", None   # raced; reload and re-decide
                run_span.set("status", "paused")
                events = await store.load(run_id)
                return "paused", Result(output="(paused: awaiting approval)",
                                        messages=state.messages, events=events)
            # decision == "granted": execute the now-approved action.

        return "execute", None
