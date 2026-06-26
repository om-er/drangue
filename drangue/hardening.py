"""The defensive tool wrapper (Chapter 6).

Every tool call goes through `run_tool`, which applies, in fixed order:

    bounded time  ->  classify failure  ->  retry transient  ->  validate  ->
    clean failure

The result is that the model never sees a raw exception or plausible garbage. It
sees either the tool's value or a structured, category-clear failure it can
reason about: try another approach, escalate, or record the avenue as closed.

A tool with no policy (the default) behaves exactly as before: call once, no
timeout, exceptions reported as a clean failure rather than crashing the run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing as t
from dataclasses import dataclass

from .errors import (
    AuthError,
    PermanentError,
    RateLimitError,
    TransientError,
    ValidationError,
)

_UNSET = object()

# Failure kinds that are retried by default.
DEFAULT_RETRY_ON: tuple = (TransientError, RateLimitError, asyncio.TimeoutError, TimeoutError)


@dataclass
class ToolPolicy:
    timeout: float | None = None        # seconds; None means no timeout
    retries: int = 0                    # extra attempts after the first
    backoff: float = 0.0                # base seconds; delay = backoff * 2**attempt
    max_backoff: float = 30.0
    retry_on: tuple = DEFAULT_RETRY_ON
    validate: t.Callable | None = None  # called on the result; may raise ValidationError
    fallback: t.Any = _UNSET            # returned (marked degraded) on terminal failure


def make_policy(*, timeout=None, retries=0, backoff=0.0, max_backoff=30.0,
                retry_on=None, validate=None, fallback=_UNSET) -> ToolPolicy:
    combined = DEFAULT_RETRY_ON + tuple(retry_on) if retry_on else DEFAULT_RETRY_ON
    return ToolPolicy(
        timeout=timeout, retries=retries, backoff=backoff, max_backoff=max_backoff,
        retry_on=combined, validate=validate, fallback=fallback,
    )


def _classify(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, AuthError):
        return "auth"
    if isinstance(exc, ValidationError):
        return "validation"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, TransientError):
        return "transient"
    if isinstance(exc, PermanentError):
        return "permanent"
    return "error"


def _delay(policy: ToolPolicy, attempt: int, exc: Exception) -> float:
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        return min(float(exc.retry_after), policy.max_backoff)
    if not policy.backoff:
        return 0.0
    return min(policy.backoff * (2 ** attempt), policy.max_backoff)


async def _invoke(tool, kwargs: dict, timeout: float | None):
    if inspect.iscoroutinefunction(tool.func):
        coro = tool.func(**kwargs)
    else:
        # Run sync tools off the event loop so they cannot block it.
        coro = asyncio.to_thread(tool.func, **kwargs)
    if timeout is not None:
        return await asyncio.wait_for(coro, timeout)
    return await coro


def _ok(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _failure(tool_name: str, category: str, message: str, policy: ToolPolicy) -> str:
    err: dict = {
        "ok": False,
        "tool": tool_name,
        "error": {"category": category, "message": message or category},
    }
    if policy.fallback is not _UNSET:
        err["degraded"] = True
        err["fallback"] = policy.fallback() if callable(policy.fallback) else policy.fallback
    return json.dumps(err, default=str)


def unknown_tool_error(name: str) -> str:
    return json.dumps({
        "ok": False,
        "tool": name,
        "error": {"category": "unknown_tool", "message": f"unknown tool '{name}'"},
    })


async def run_tool(tool, kwargs: dict, *, sleep=asyncio.sleep) -> str:
    """Invoke a tool under its policy and return content for the model."""
    policy = tool.policy or ToolPolicy()
    attempt = 0
    while True:
        try:
            result = await _invoke(tool, kwargs, policy.timeout)
            if policy.validate is not None:
                result = policy.validate(result)   # may raise ValidationError
            return _ok(result)
        except Exception as exc:  # noqa: BLE001 - classify, never propagate
            category = _classify(exc)
            if isinstance(exc, policy.retry_on) and attempt < policy.retries:
                await sleep(_delay(policy, attempt, exc))   # same kwargs -> same idempotency key
                attempt += 1
                continue
            return _failure(tool.name, category, str(exc), policy)
