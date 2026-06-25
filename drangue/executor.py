"""The executor: side-effecting, non-deterministic step execution.

This is where the model gets called and tools get run. Everything the
orchestrator must stay away from (I/O, latency, failure) lives here. Each
executed step returns the Event that records what happened, which the engine
appends to the log.

The idempotency_key is threaded through now and used for real in Phase 2, where
retries reuse it so a side effect runs exactly once across a crash.
"""

from __future__ import annotations

import asyncio
import inspect
import json

from .events import Event
from .orchestrator import ModelStep, ToolStep


class Executor:
    def __init__(self, model, tools: dict):
        self.model = model
        self.tools = tools

    async def run(self, step, *, system: str, messages: list, idempotency_key: str) -> Event:
        if isinstance(step, ModelStep):
            resp = await self.model.generate(
                system=system,
                messages=messages,
                tools=list(self.tools.values()),
            )
            return Event(seq=step.seq, type="model_decision", payload={
                "text": resp.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in resp.tool_calls
                ],
                "usage": resp.usage,
            })

        if isinstance(step, ToolStep):
            content = await self._dispatch(step.call)
            return Event(seq=step.seq, type="tool_result", payload={
                "call_id": step.call.id,
                "name": step.call.name,
                "content": content,
            })

        raise TypeError(f"Unknown step: {step!r}")

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
