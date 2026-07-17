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
import random
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

# Failure kinds that are retried by default. Timeouts are deliberately NOT
# here: a timed-out sync tool's thread may still be running (see _invoke), so
# retrying a timeout can execute a non-idempotent side effect twice,
# concurrently. Opt in with retry_on=DEFAULT_RETRY_ON + (TimeoutError,) for
# tools where that is safe.
DEFAULT_RETRY_ON: tuple = (TransientError, RateLimitError)

DEFAULT_BACKOFF = 0.5  # base seconds; instant retries hammer a struggling backend


@dataclass
class ToolPolicy:
    timeout: float | None = None        # seconds; None means no timeout
    retries: int = 0                    # extra attempts after the first
    backoff: float = DEFAULT_BACKOFF    # base seconds; delay = backoff * 2**attempt, jittered
    max_backoff: float = 30.0
    retry_on: tuple = DEFAULT_RETRY_ON
    validate: t.Callable | None = None  # called on the result; may raise ValidationError
    fallback: t.Any = _UNSET            # returned (marked degraded) on terminal failure


def make_policy(*, timeout=None, retries=0, backoff=DEFAULT_BACKOFF,
                max_backoff=30.0, retry_on=None, validate=None,
                fallback=_UNSET) -> ToolPolicy:
    # An explicit retry_on REPLACES the defaults, so a policy can also narrow
    # what is retried (e.g. never retry a non-idempotent tool on rate limits).
    return ToolPolicy(
        timeout=timeout, retries=retries, backoff=backoff, max_backoff=max_backoff,
        retry_on=tuple(retry_on) if retry_on is not None else DEFAULT_RETRY_ON,
        validate=validate, fallback=fallback,
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


def _delay(policy: ToolPolicy, attempt: int, exc: Exception,
           rand: t.Callable[[], float] = random.random) -> float:
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        # A server-mandated wait is honored as given; clamping it to our own
        # backoff cap would just guarantee another rejection.
        return max(0.0, float(exc.retry_after))
    if not policy.backoff:
        return 0.0
    ceiling = min(policy.backoff * (2 ** attempt), policy.max_backoff)
    # Equal jitter: half the window is deterministic, half random, so replicas
    # that failed together do not retry in synchronized bursts.
    return ceiling * (0.5 + 0.5 * rand())


async def _invoke(tool, kwargs: dict, timeout: float | None):
    if inspect.iscoroutinefunction(tool.func):
        coro = tool.func(**kwargs)
    else:
        # Run sync tools off the event loop so they cannot block it.
        coro = asyncio.to_thread(tool.func, **kwargs)
    if timeout is not None:
        # Caveat: for a SYNC tool, wait_for cancels the waiter but the worker
        # thread keeps running to completion (Python cannot cancel threads). So
        # the timeout bounds the caller, not the side effect, and a retry may run
        # concurrently with the orphaned first attempt. Truly bounded execution
        # needs a cancellable async tool or an out-of-process worker.
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
        # The fallback itself may fail (a cache lookup, say); that must not
        # break the "classify, never propagate" contract of run_tool.
        try:
            value = policy.fallback() if callable(policy.fallback) else policy.fallback
        except Exception as fexc:  # noqa: BLE001
            err["error"]["fallback_error"] = str(fexc) or type(fexc).__name__
        else:
            err["degraded"] = True
            err["fallback"] = value
    return json.dumps(err, default=str)


def unknown_tool_error(name: str) -> str:
    return json.dumps({
        "ok": False,
        "tool": name,
        "error": {"category": "unknown_tool", "message": f"unknown tool '{name}'"},
    })


async def run_tool(tool, kwargs: dict, *, sleep=asyncio.sleep,
                   rand: t.Callable[[], float] = random.random) -> str:
    """Invoke a tool under its policy and return content for the model."""
    policy = tool.policy or ToolPolicy()
    attempt = 0
    while True:
        try:
            result = await _invoke(tool, kwargs, policy.timeout)
            if policy.validate is not None:
                # Check-style validators inspect and return None (keep the
                # result); transform-style validators return the replacement.
                # Either style raises ValidationError to reject.
                checked = policy.validate(result)
                if checked is not None:
                    result = checked
            return _ok(result)
        except Exception as exc:  # noqa: BLE001 - classify, never propagate
            category = _classify(exc)
            if isinstance(exc, policy.retry_on) and attempt < policy.retries:
                await sleep(_delay(policy, attempt, exc, rand))   # same kwargs -> same idempotency key
                attempt += 1
                continue
            return _failure(tool.name, category, str(exc), policy)
