"""drangue-memory tests: conformance + end-to-end inside an agent. Offline."""

import os
import tempfile

from drangue import Agent, MemoryItem, ModelResponse, tool
from drangue.models import Model
from drangue.testing import check_memory

from drangue_memory import SqliteMemory


@tool
def noop(x: str) -> str:
    """A read-only no-op tool."""
    return "ok"


class SystemCapturingModel(Model):
    def __init__(self):
        self.systems = []

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.systems.append(system)
        return ModelResponse(text="done")


def _tmp():
    return os.path.join(tempfile.mkdtemp(), "mem.db")


async def test_sqlite_memory_conforms():
    paths, mems = [], []

    def make():
        p = _tmp()
        paths.append(p)
        mem = SqliteMemory(p)
        mems.append(mem)
        return mem

    try:
        await check_memory(make)
    finally:
        # SqliteMemory holds its connection open, and Windows refuses to unlink
        # a file that still has one. Close before removing.
        for m in mems:
            m.close()
        for p in paths:
            if os.path.exists(p):
                os.remove(p)


async def test_remembered_item_is_recalled_into_the_agent():
    path = _tmp()
    try:
        mem = SqliteMemory(path)
        await mem.remember(MemoryItem(
            key="inc-1",
            value="payments latency was caused by a downstream timeout"))

        model = SystemCapturingModel()
        agent = Agent(model=model, tools=[noop], instructions="You are an SRE.",
                      memory=mem)
        result = await agent.run("investigate payments latency", run_id="r")

        assert "downstream timeout" in model.systems[0]
        assert any(e.type == "memory_recalled" for e in result.events)
        mem.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


async def test_expired_item_is_dropped_by_the_runtime():
    path = _tmp()
    try:
        mem = SqliteMemory(path)
        # An item that expired long ago; recall surfaces it, the engine drops it.
        await mem.remember(MemoryItem(key="old", value="payments stale fact",
                                      expires_at=1.0))
        model = SystemCapturingModel()
        agent = Agent(model=model, tools=[noop], memory=mem)
        result = await agent.run("payments", run_id="r")

        recalled = [e for e in result.events if e.type == "memory_recalled"][0]
        assert recalled.payload["items"] == []
        assert "stale fact" not in model.systems[0]
        mem.close()
    finally:
        if os.path.exists(path):
            os.remove(path)
