"""Phase 3 tests: tool hardening (Chapter 6).

Unit tests drive `run_tool` directly with a fake sleep so backoff is asserted
without real delay. Integration tests go through the agent to confirm clean
failures reach the model and the run still completes. All offline.
"""

import json

from drangue import (
    Agent,
    ModelResponse,
    PermanentError,
    RateLimitError,
    ToolCall,
    TransientError,
    ValidationError,
    tool,
)
from drangue.hardening import run_tool
from drangue.models import Model


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    async def generate(self, *, system, messages, tools):
        resp = self.steps[self.i]
        self.i += 1
        return resp


async def _no_sleep(_):
    pass


async def test_transient_failure_is_retried_then_succeeds():
    calls = {"n": 0}

    @tool(retries=3, backoff=1.0)
    def flaky() -> str:
        """Flaky tool."""
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("blip")
        return "recovered"

    delays = []

    async def fake_sleep(d):
        delays.append(d)

    out = await run_tool(flaky, {}, sleep=fake_sleep)
    assert out == "recovered"
    assert calls["n"] == 3
    assert delays == [1.0, 2.0]   # exponential backoff between the two retries


async def test_permanent_failure_is_not_retried():
    calls = {"n": 0}

    @tool(retries=3)
    def broken() -> str:
        """Always broken."""
        calls["n"] += 1
        raise PermanentError("gone")

    out = await run_tool(broken, {}, sleep=_no_sleep)
    assert calls["n"] == 1        # not retried
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert parsed["error"]["category"] == "permanent"


async def test_rate_limit_respects_retry_after():
    n = {"i": 0}

    @tool(retries=2)
    def hit() -> str:
        """Rate-limited tool."""
        n["i"] += 1
        if n["i"] == 1:
            raise RateLimitError("slow down", retry_after=2.0)
        return "ok"

    delays = []

    async def fake_sleep(d):
        delays.append(d)

    out = await run_tool(hit, {}, sleep=fake_sleep)
    assert out == "ok"
    assert delays == [2.0]        # honored Retry-After instead of computed backoff


async def test_timeout_is_a_clean_failure():
    import asyncio

    @tool(timeout=0.01)
    async def slow() -> str:
        """Slow tool."""
        await asyncio.sleep(1.0)
        return "done"

    out = await run_tool(slow, {}, sleep=_no_sleep)
    parsed = json.loads(out)
    assert parsed["error"]["category"] == "timeout"


async def test_validation_failure_returns_clean_error():
    def reject(_result):
        raise ValidationError("unexpected shape")

    @tool(validate=reject)
    def fetch() -> str:
        """Fetch raw data."""
        return "raw bytes that do not match the contract"

    out = await run_tool(fetch, {}, sleep=_no_sleep)
    parsed = json.loads(out)
    assert parsed["error"]["category"] == "validation"


async def test_fallback_is_returned_marked_degraded():
    @tool(fallback="cached value")
    def lookup() -> str:
        """Look something up."""
        raise PermanentError("backend down")

    out = await run_tool(lookup, {}, sleep=_no_sleep)
    parsed = json.loads(out)
    assert parsed["degraded"] is True
    assert parsed["fallback"] == "cached value"


async def test_clean_failure_reaches_model_and_run_completes():
    @tool(retries=1, backoff=0.0)
    def boom(city: str) -> str:
        """Always fails."""
        raise TransientError("upstream down")

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "boom", {"city": "X"})]),
        ModelResponse(text="handled gracefully"),
    ])
    result = await Agent(model=model, tools=[boom]).run("go")

    assert result.output == "handled gracefully"
    tool_result = [e for e in result.events if e.type == "tool_result"][0]
    parsed = json.loads(tool_result.payload["content"])
    assert parsed["ok"] is False
    assert parsed["error"]["category"] == "transient"


async def test_unknown_tool_is_a_clean_structured_error():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "ghost", {})]),
        ModelResponse(text="ok"),
    ])
    result = await Agent(model=model, tools=[]).run("go")

    tool_result = [e for e in result.events if e.type == "tool_result"][0]
    parsed = json.loads(tool_result.payload["content"])
    assert parsed["error"]["category"] == "unknown_tool"
