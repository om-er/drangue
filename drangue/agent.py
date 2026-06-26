"""The agent: a small facade over a durable orchestrator and executor.

The simple path stays simple:

    agent = Agent("claude-opus-4-8", tools=[get_weather])
    result = agent.run_sync("What should I wear in Paris?")

Underneath, the run is driven as an event-sourced loop: the orchestrator decides
each step deterministically, the executor performs it, and every step is
appended to a store. In Phase 0 the store is in-memory; swapping it for a
durable one (Phase 2) makes runs resumable with no change to this facade.
"""

from __future__ import annotations

import asyncio
import typing as t
import uuid

from . import rollout
from .engine import EventSourcedEngine
from .events import Event, Result
from .executor import Executor
from .models import AnthropicModel
from .observability import ConsoleTracer, NullTracer
from .orchestrator import Orchestrator
from .routing import SingleModel
from .store import InMemoryStore
from .tool import Tool
from .tool import tool as make_tool


def _resolve_model(model: t.Any, max_tokens: int):
    if isinstance(model, str):
        return AnthropicModel(model, max_tokens=max_tokens)
    return model  # assume it implements the Model interface


def _as_tool(obj: t.Any) -> Tool:
    if isinstance(obj, Tool):
        return obj
    if callable(obj):
        return make_tool(obj)
    raise TypeError(f"Expected a tool or callable, got {type(obj)!r}")


def _new_run_id() -> str:
    return uuid.uuid4().hex


class Agent:
    """A model plus tools. Call `run` / `stream` (async) or `run_sync`."""

    def __init__(self, model: t.Any = None, tools: list | None = None,
                 instructions: str = "", *, max_steps: int = 20,
                 max_tokens: int = 4096, store=None, engine=None, tracer=None,
                 router=None, budget=None, guardrails=None, autonomy=None):
        self.router = self._resolve_router(model, router, max_tokens)
        self.tools: dict[str, Tool] = {}
        for obj in tools or []:
            tool = _as_tool(obj)
            self.tools[tool.name] = tool
        self.instructions = instructions
        self.store = store or InMemoryStore()
        self.engine = engine or EventSourcedEngine()
        self.tracer = tracer or NullTracer()
        self.budget = budget
        self.guardrails = guardrails
        self.autonomy = autonomy
        self.orchestrator = Orchestrator(max_steps=max_steps)
        self.executor = Executor(self.router, self.tools, guardrails=guardrails)

    @staticmethod
    def _resolve_router(model, router, max_tokens):
        if router is not None:
            return router
        if model is not None and hasattr(model, "choose"):
            return model  # a Router was passed as `model`
        return SingleModel(_resolve_model(model, max_tokens))

    def _tracer_for(self, trace: bool):
        return ConsoleTracer() if trace else self.tracer

    async def run(self, input: str | None = None, *, run_id: str | None = None,
                  trace: bool = False) -> Result:
        """Run to completion, or resume a paused/crashed run by run_id."""
        rid = run_id or _new_run_id()
        return await self.engine.run(
            run_id=rid, orchestrator=self.orchestrator, executor=self.executor,
            store=self.store, system=self.instructions, input=input or "",
            tracer=self._tracer_for(trace), budget=self.budget,
            autonomy=self.autonomy,
        )

    async def resume(self, run_id: str, *, trace: bool = False) -> Result:
        """Resume a run (e.g. after an approval). Alias for run with a run_id."""
        return await self.run(run_id=run_id, trace=trace)

    async def pending_approvals(self, run_id: str) -> list:
        """Actions awaiting a human decision, each with the agent's reasoning."""
        events = await self.store.load(run_id)
        return rollout.pending_approvals(events)

    async def approve(self, run_id: str, call_id: str | None = None) -> None:
        """Approve a pending action. Defaults to the one pending approval."""
        events = await self.store.load(run_id)
        cid = call_id or rollout.first_pending_call(events)
        await self.store.append(run_id, Event(
            seq=rollout.next_seq(events), type="approval_granted",
            payload={"call_id": cid}))

    async def reject(self, run_id: str, call_id: str | None = None,
                     reason: str = "") -> None:
        """Reject a pending action; the agent is told and continues."""
        events = await self.store.load(run_id)
        cid = call_id or rollout.first_pending_call(events)
        await self.store.append(run_id, Event(
            seq=rollout.next_seq(events), type="approval_denied",
            payload={"call_id": cid, "reason": reason}))

    async def stream(self, input: str, *, run_id: str | None = None,
                     trace: bool = False) -> t.AsyncIterator[Event]:
        """Yield each Event as it is appended to the log."""
        rid = run_id or _new_run_id()
        tracer = self._tracer_for(trace)
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def emit(ev: Event) -> None:
            await queue.put(ev)

        async def driver():
            try:
                return await self.engine.run(
                    run_id=rid, orchestrator=self.orchestrator,
                    executor=self.executor, store=self.store,
                    system=self.instructions, input=input, emit=emit,
                    tracer=tracer, budget=self.budget, autonomy=self.autonomy,
                )
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(driver())
        try:
            while True:
                ev = await queue.get()
                if ev is sentinel:
                    break
                yield ev
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def run_sync(self, input: str | None = None, *, run_id: str | None = None,
                 trace: bool = False) -> Result:
        """Convenience wrapper for scripts not already in an event loop."""
        return asyncio.run(self.run(input, run_id=run_id, trace=trace))
