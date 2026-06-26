"""The executor: side-effecting, non-deterministic step execution.

This is where the model gets called and tools get run. Everything the
orchestrator must stay away from (I/O, latency, failure, the clock) lives here.
Each executed step returns the Event that records what happened, stamped with
timing, which the engine appends to the log. Reading the clock here is fine: the
executor is non-deterministic by design, and its results become recorded facts
that replay reads back rather than re-computing.

The idempotency_key is threaded through now and used for real in Phase 2, where
retries reuse it so a side effect runs exactly once across a crash.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time

from .events import Event
from .orchestrator import ModelStep, ToolStep


class Executor:
    def __init__(self, model, tools: dict):
        self.model = model
        self.tools = tools

    async def run(self, step, *, system: str, messages: list,
                  idempotency_key: str, tracer) -> Event:
        if isinstance(step, ModelStep):
            return await self._run_model(step, system=system, messages=messages, tracer=tracer)
        if isinstance(step, ToolStep):
            return await self._run_tool(step, tracer=tracer)
        raise TypeError(f"Unknown step: {step!r}")

    async def _run_model(self, step, *, system, messages, tracer) -> Event:
        t0 = time.monotonic()
        with tracer.span("model", seq=step.seq) as span:
            resp = await self.model.generate(
                system=system, messages=messages, tools=list(self.tools.values()),
            )
            duration = (time.monotonic() - t0) * 1000.0
            tool_calls = [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in resp.tool_calls
            ]
            span.set("text", resp.text)
            span.set("tool_calls", tool_calls)
            span.set("usage", resp.usage)
            span.set("reasoning", resp.reasoning)
            span.set("duration_ms", duration)

        return Event(
            seq=step.seq, type="model_decision",
            ts=time.time(), duration_ms=duration,
            payload={
                "text": resp.text,
                "tool_calls": tool_calls,
                "usage": resp.usage,
                "reasoning": resp.reasoning,
            },
        )

    async def _run_tool(self, step, *, tracer) -> Event:
        call = step.call
        t0 = time.monotonic()
        with tracer.span("tool", seq=step.seq, name=call.name,
                         arguments=call.arguments) as span:
            content = await self._dispatch(call)
            duration = (time.monotonic() - t0) * 1000.0
            span.set("content", content)
            span.set("duration_ms", duration)

        return Event(
            seq=step.seq, type="tool_result",
            ts=time.time(), duration_ms=duration,
            payload={"call_id": call.id, "name": call.name, "content": content},
        )

    async def _dispatch(self, call) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"Error: unknown tool '{call.name}'"
        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**call.arguments)
            else:
                # Run sync tools off the event loop so they cannot block it.
                result = await asyncio.to_thread(tool.func, **call.arguments)
        except Exception as exc:  # clean failure over a crashed run
            return f"Error: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)
