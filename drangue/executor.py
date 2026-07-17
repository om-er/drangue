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
import time

from .errors import RateLimitError, TransientError
from .events import Event
from .guardrails import blocked_result
from .hardening import (
    DEFAULT_BACKOFF,
    MALFORMED_ARGS_KEY,
    ToolPolicy,
    _delay,
    malformed_arguments_error,
    run_tool,
    unknown_tool_error,
)
from .orchestrator import ModelStep, ToolStep


class Executor:
    def __init__(self, router, tools: dict, guardrails=None, *,
                 model_retries: int = 2):
        self.router = router
        self.tools = tools
        self.guardrails = guardrails
        # Retry policy for the model call itself. It fires only on drangue's
        # own transient categories (TransientError / RateLimitError, which a
        # custom Model or wrapper raises deliberately); provider SDK errors
        # pass through untouched — the SDKs already retry those internally.
        self.model_policy = ToolPolicy(retries=model_retries,
                                       backoff=DEFAULT_BACKOFF,
                                       retry_on=(TransientError, RateLimitError))

    async def check_input(self, text: str) -> str | None:
        if self.guardrails is None:
            return None
        return await self.guardrails.check_input(text)

    async def run(self, step, *, system: str, messages: list,
                  idempotency_key: str, tracer) -> Event:
        if isinstance(step, ModelStep):
            return await self._run_model(step, system=system, messages=messages,
                                         idempotency_key=idempotency_key, tracer=tracer)
        if isinstance(step, ToolStep):
            return await self._run_tool(step, idempotency_key=idempotency_key, tracer=tracer)
        raise TypeError(f"Unknown step: {step!r}")

    async def _run_model(self, step, *, system, messages, idempotency_key, tracer) -> Event:
        model = self.router.choose(messages=messages, step_index=step.index)
        model_name = getattr(model, "model", None)
        started = time.time()   # Event.ts is the step's start, by contract
        t0 = time.monotonic()
        with tracer.span("model", seq=step.seq, model=model_name) as span:
            attempt = 0
            while True:
                try:
                    resp = await model.generate(
                        system=system, messages=messages,
                        tools=list(self.tools.values()),
                        idempotency_key=idempotency_key,
                    )
                    break
                except self.model_policy.retry_on as exc:
                    if attempt >= self.model_policy.retries:
                        raise
                    # Same idempotency key across attempts: a backend that
                    # honors it returns the first response instead of charging
                    # twice.
                    await asyncio.sleep(_delay(self.model_policy, attempt, exc))
                    attempt += 1
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
            ts=started, duration_ms=duration,
            payload={
                "text": resp.text,
                "tool_calls": tool_calls,
                "usage": resp.usage,
                "reasoning": resp.reasoning,
                "model": model_name,
                "stop_reason": resp.stop_reason,
            },
        )

    async def _run_tool(self, step, *, idempotency_key: str, tracer) -> Event:
        call = step.call
        started = time.time()   # Event.ts is the step's start, by contract
        t0 = time.monotonic()
        with tracer.span("tool", seq=step.seq, name=call.name,
                         arguments=call.arguments) as span:
            content = await self._dispatch(call, idempotency_key)
            duration = (time.monotonic() - t0) * 1000.0
            span.set("content", content)
            span.set("duration_ms", duration)

        return Event(
            seq=step.seq, type="tool_result",
            ts=started, duration_ms=duration,
            payload={"call_id": call.id, "name": call.name, "content": content},
        )

    async def _dispatch(self, call, idempotency_key: str) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            return unknown_tool_error(call.name)
        if MALFORMED_ARGS_KEY in call.arguments:
            # The adapter could not parse what the model sent; report it as a
            # clean failure the model can correct instead of invoking the tool.
            return malformed_arguments_error(call.name, call.arguments[MALFORMED_ARGS_KEY])
        # Enforce guardrails in code, regardless of what the model decided.
        if self.guardrails is not None:
            reason = await self.guardrails.check_tool(tool, call.arguments)
            if reason is not None:
                return blocked_result(tool.name, reason)
        kwargs = dict(call.arguments)
        if tool.wants_idempotency_key:
            # A stable key derived from durable facts (run_id, seq). A tool that
            # declares it can pass it downstream to deduplicate side effects.
            # The same kwargs are reused across retries, so the key is stable.
            kwargs["idempotency_key"] = idempotency_key
        # run_tool applies the tool's policy: timeout, retry, validate, and turns
        # any failure into a clean, structured result the model can reason about.
        return await run_tool(tool, kwargs)
