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

from .engine import EventSourcedEngine
from .events import Event, Result
from .executor import Executor
from .models import AnthropicModel
from .observability import ConsoleTracer, NullTracer
from .orchestrator import Orchestrator
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

    def __init__(self, model: t.Any, tools: list | None = None,
                 instructions: str = "", *, max_steps: int = 20,
                 max_tokens: int = 4096, store=None, engine=None, tracer=None):
        self.model = _resolve_model(model, max_tokens)
        self.tools: dict[str, Tool] = {}
        for obj in tools or []:
            tool = _as_tool(obj)
            self.tools[tool.name] = tool
        self.instructions = instructions
        self.store = store or InMemoryStore()
        self.engine = engine or EventSourcedEngine()
        self.tracer = tracer or NullTracer()
        self.orchestrator = Orchestrator(max_steps=max_steps)
        self.executor = Executor(self.model, self.tools)

    def _tracer_for(self, trace: bool):
        return ConsoleTracer() if trace else self.tracer

    async def run(self, input: str, *, run_id: str | None = None,
                  trace: bool = False) -> Result:
        """Run to completion. Pass an existing run_id to resume (Phase 2)."""
        rid = run_id or _new_run_id()
        return await self.engine.run(
            run_id=rid, orchestrator=self.orchestrator, executor=self.executor,
            store=self.store, system=self.instructions, input=input,
            tracer=self._tracer_for(trace),
        )

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
                    tracer=tracer,
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

    def run_sync(self, input: str, *, run_id: str | None = None,
                 trace: bool = False) -> Result:
        """Convenience wrapper for scripts not already in an event loop."""
        return asyncio.run(self.run(input, run_id=run_id, trace=trace))
