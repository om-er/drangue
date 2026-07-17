"""Parallel sibling tool calls: concurrent when safe, serial when not.

The concurrency proof uses two tools that each wait for the other to start;
they can only both finish if they genuinely overlap. All offline.
"""

import asyncio
import json

from drangue import Agent, Autonomy, ModelResponse, ToolCall, tool
from drangue.models import Model


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        resp = self.steps[self.i]
        self.i += 1
        return resp


def _rendezvous_tools():
    ev_a, ev_b = asyncio.Event(), asyncio.Event()

    @tool
    async def left() -> str:
        """Waits for right to start."""
        ev_a.set()
        await asyncio.wait_for(ev_b.wait(), timeout=5)
        return "left done"

    @tool
    async def right() -> str:
        """Waits for left to start."""
        ev_b.set()
        await asyncio.wait_for(ev_a.wait(), timeout=5)
        return "right done"

    return left, right


async def test_sibling_calls_run_concurrently_and_land_in_call_order():
    left, right = _rendezvous_tools()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "left", {}),
                                  ToolCall("c2", "right", {})]),
        ModelResponse(text="both done"),
    ])
    result = await Agent(model=model, tools=[left, right]).run("go")

    assert result.output == "both done"
    tool_results = [e for e in result.events if e.type == "tool_result"]
    # Deterministic order: results appended by tool-call order, consecutive seqs.
    assert [tr.payload["call_id"] for tr in tool_results] == ["c1", "c2"]
    assert [tr.payload["content"] for tr in tool_results] == ["left done", "right done"]
    assert tool_results[1].seq == tool_results[0].seq + 1


async def test_parallel_run_replays_without_reexecution():
    from drangue import InMemoryStore

    left, right = _rendezvous_tools()
    store = InMemoryStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "left", {}),
                                  ToolCall("c2", "right", {})]),
        ModelResponse(text="both done"),
    ])
    agent = Agent(model=model, tools=[left, right], store=store)
    await agent.run("go", run_id="p1")

    again = await agent.run("go", run_id="p1")
    assert again.output == "both done"
    assert model.i == 2                      # nothing re-ran on replay


async def test_an_assisted_sibling_forces_the_serial_path_and_pauses():
    ran = []

    @tool
    def safe(x: int) -> str:
        """Autonomous tool."""
        ran.append(("safe", x))
        return "ok"

    @tool
    def risky(x: int) -> str:
        """Assisted tool."""
        ran.append(("risky", x))
        return "did it"

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "safe", {"x": 1}),
                                  ToolCall("c2", "risky", {"x": 2})]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[safe, risky],
                  autonomy=Autonomy(modes={"risky": "assisted"}))

    paused = await agent.run("go", run_id="mix")
    assert paused.status == "paused"
    assert ran == [("safe", 1)]              # serial: safe ran, risky paused

    await agent.approve("mix")
    result = await agent.resume("mix")
    assert result.output == "done"
    assert ran == [("safe", 1), ("risky", 2)]


async def test_an_irreversible_sibling_forces_the_serial_intent_path():
    ran = []

    @tool
    def read(x: int) -> str:
        """Reversible."""
        ran.append("read")
        return "data"

    @tool(reversible=False)
    def wire(x: int) -> str:
        """Irreversible."""
        ran.append("wire")
        return "sent"

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "read", {"x": 1}),
                                  ToolCall("c2", "wire", {"x": 2})]),
        ModelResponse(text="done"),
    ])
    result = await Agent(model=model, tools=[read, wire]).run("go")

    assert result.output == "done"
    assert sorted(ran) == ["read", "wire"]
    # The irreversible call still got its intent marker.
    intents = [e for e in result.events if e.type == "step_started"]
    assert [i.payload["call_id"] for i in intents] == ["c2"]


async def test_a_failing_sibling_is_a_clean_result_not_a_crash():
    left, right = _rendezvous_tools()

    @tool
    async def boom() -> str:
        """Fails while the others succeed."""
        raise RuntimeError("upstream down")

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "left", {}),
                                  ToolCall("c2", "right", {}),
                                  ToolCall("c3", "boom", {})]),
        ModelResponse(text="handled"),
    ])
    result = await Agent(model=model, tools=[left, right, boom]).run("go")

    assert result.output == "handled"
    contents = {e.payload["call_id"]: e.payload["content"]
                for e in result.events if e.type == "tool_result"}
    assert contents["c1"] == "left done"
    assert json.loads(contents["c3"])["ok"] is False
