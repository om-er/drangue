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
from .context import RunContext
from .engine import EventSourcedEngine
from .errors import ConflictError, UnknownRunError
from .events import Event, Result
from .executor import Executor
from .models import AnthropicModel
from .observability import ConsoleTracer, NullTracer
from .orchestrator import Orchestrator
from .routing import SingleModel
from .store import InMemoryStore
from .tool import Tool
from .tool import tool as make_tool


def _resolve_model(model: t.Any, max_tokens: int, cache: bool = False):
    if isinstance(model, str):
        return AnthropicModel(model, max_tokens=max_tokens, cache=cache)
    if cache:
        # A prebuilt model carries its own cache setting; silently dropping this
        # one would leave caching looking enabled while it is not.
        raise ValueError(
            "cache=True only applies when the model is named as a string. "
            "You passed a model instance, so configure it there instead: "
            "AnthropicModel(..., cache=True)."
        )
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
    """A model plus tools. Call `run` / `stream` (async) or `run_sync`.

    `cache=True` marks the stable prefix (system prompt and tool schemas) so the
    provider can reuse it across the steps of a run. It applies only when the
    model is named as a string, since that is the only case where this facade
    builds the model; with a model instance or a router, set it there.
    """

    def __init__(self, model: t.Any = None, tools: list | None = None,
                 instructions: str = "", *, max_steps: int = 20,
                 max_tokens: int = 4096, cache: bool = False, store=None,
                 engine=None, tracer=None, router=None, budget=None,
                 guardrails=None, autonomy=None, memory=None,
                 model_retries: int = 2):
        self.router = self._resolve_router(model, router, max_tokens, cache)
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
        self.memory = memory
        self.orchestrator = Orchestrator(max_steps=max_steps)
        self.executor = Executor(self.router, self.tools, guardrails=guardrails,
                                 model_retries=model_retries)

    @staticmethod
    def _resolve_router(model, router, max_tokens, cache=False):
        explicit = router
        if explicit is None and model is not None and hasattr(model, "choose"):
            explicit = model  # a Router was passed as `model`
        if explicit is not None:
            if cache:
                # The router owns the models, so this flag would go nowhere.
                raise ValueError(
                    "cache=True only applies when the model is named as a "
                    "string. With a router, set cache on the models it "
                    "returns: AnthropicModel(..., cache=True)."
                )
            return explicit
        return SingleModel(_resolve_model(model, max_tokens, cache))

    def _tracer_for(self, trace: bool):
        return ConsoleTracer() if trace else self.tracer

    def _context(self, input, run_id, *, trace=False, emit=None) -> RunContext:
        return RunContext(
            run_id=run_id or _new_run_id(),
            input=input or "",
            orchestrator=self.orchestrator,
            executor=self.executor,
            store=self.store,
            system=self.instructions,
            tracer=self._tracer_for(trace),
            emit=emit,
            budget=self.budget,
            autonomy=self.autonomy,
            memory=self.memory,
        )

    async def run(self, input: str | None = None, *, run_id: str | None = None,
                  trace: bool = False) -> Result:
        """Run to completion, or resume a paused/crashed run by run_id."""
        return await self.engine.run(self._context(input, run_id, trace=trace))

    async def resume(self, run_id: str, *, trace: bool = False) -> Result:
        """Resume a run (e.g. after an approval). Alias for run with a run_id."""
        if not await self.store.load(run_id):
            # A typo'd run_id must not silently start a brand-new empty run.
            raise UnknownRunError(
                f"run {run_id!r} not found in the store; resume replays an "
                "existing log, it cannot start one"
            )
        return await self.run(run_id=run_id, trace=trace)

    async def remember(self, item) -> None:
        """Write to long-term memory. Deliberate, not automatic: store what is
        worth recalling later, not every run."""
        if self.memory is not None:
            await self.memory.remember(item)

    async def pending_approvals(self, run_id: str) -> list:
        """Actions awaiting a human decision, each with the agent's reasoning."""
        events = await self.store.load(run_id)
        return rollout.pending_approvals(events)

    @staticmethod
    def _resolve_pending(events, call_id, run_id) -> str:
        """The pending call a decision applies to; raise rather than record junk."""
        pending = [p["call_id"] for p in rollout.pending_approvals(events)]
        if call_id is None:
            if not pending:
                raise ValueError(f"run {run_id!r} has no pending approval to decide")
            return pending[0]
        if call_id not in pending:
            raise ValueError(
                f"call {call_id!r} is not pending approval in run {run_id!r} "
                f"(pending: {pending})"
            )
        return call_id

    async def _decide(self, run_id: str, call_id: str | None, type_: str,
                      extra: dict) -> None:
        # Recording a decision is read-modify-write on the log's next seq; a
        # concurrent engine step can claim it first. On conflict, reload and
        # retry at the new tail instead of losing the human's decision.
        last: Exception | None = None
        for _ in range(5):
            events = await self.store.load(run_id)
            cid = self._resolve_pending(events, call_id, run_id)
            try:
                await self.store.append(run_id, Event(
                    seq=rollout.next_seq(events), type=type_,
                    payload={"call_id": cid, **extra}))
                return
            except ConflictError as exc:
                last = exc
        raise last

    async def approve(self, run_id: str, call_id: str | None = None) -> None:
        """Approve a pending action. Defaults to the one pending approval."""
        await self._decide(run_id, call_id, "approval_granted", {})

    async def reject(self, run_id: str, call_id: str | None = None,
                     reason: str = "") -> None:
        """Reject a pending action; the agent is told and continues."""
        await self._decide(run_id, call_id, "approval_denied", {"reason": reason})

    async def stream(self, input: str | None = None, *, run_id: str | None = None,
                     trace: bool = False) -> t.AsyncIterator[Event]:
        """Yield each Event as it is appended to the log."""
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def emit(ev: Event) -> None:
            await queue.put(ev)

        ctx = self._context(input, run_id, trace=trace, emit=emit)

        async def driver():
            try:
                return await self.engine.run(ctx)
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
