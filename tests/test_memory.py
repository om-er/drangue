"""Tests for memory wired into the loop (Chapter 5, the core hook).

Recall is automatic and recorded; writing is explicit. All offline.
"""

from drangue import Agent, MemoryItem, ModelResponse, NullMemory, tool
from drangue.memory import Memory
from drangue.models import Model


@tool
def search(q: str) -> str:
    """Search."""
    return "result"


class FakeMemory(Memory):
    def __init__(self, items=None):
        self.items = items or []
        self.recalls = 0
        self.remembered = []

    async def recall(self, query, *, limit=5):
        self.recalls += 1
        return self.items

    async def remember(self, item):
        self.remembered.append(item)


class SystemCapturingModel(Model):
    """Records the system prompt it was called with, then answers."""

    def __init__(self):
        self.systems = []

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.systems.append(system)
        return ModelResponse(text="done")


async def test_recall_injects_context_into_the_system_prompt():
    mem = FakeMemory([MemoryItem(key="past", value="payments timed out last week")])
    model = SystemCapturingModel()
    agent = Agent(model=model, tools=[search], instructions="You are an SRE.",
                  memory=mem)

    result = await agent.run("investigate payments", run_id="r")

    assert mem.recalls == 1
    assert "payments timed out last week" in model.systems[0]
    assert "You are an SRE." in model.systems[0]
    assert any(e.type == "memory_recalled" for e in result.events)


async def test_recall_happens_once_and_is_replayed():
    mem = FakeMemory([MemoryItem(key="k", value="v")])
    agent = Agent(model=SystemCapturingModel(), tools=[search], memory=mem)

    await agent.run("q", run_id="same")
    assert mem.recalls == 1

    # Re-running the same completed run replays the recorded recall; no re-query.
    await agent.run("q", run_id="same")
    assert mem.recalls == 1


async def test_no_memory_means_no_recall_and_unchanged_system():
    model = SystemCapturingModel()
    agent = Agent(model=model, tools=[search])   # no memory

    result = await agent.run("q")

    assert not any(e.type == "memory_recalled" for e in result.events)
    assert model.systems[0] == ""   # instructions default; nothing appended


async def test_remember_is_explicit():
    mem = FakeMemory()
    agent = Agent(model=SystemCapturingModel(), tools=[search], memory=mem)

    await agent.remember(MemoryItem(key="incident-1", value="root cause: bad index"))

    assert len(mem.remembered) == 1
    assert mem.remembered[0].value == "root cause: bad index"


async def test_expired_memory_is_dropped_at_recall():
    mem = FakeMemory([
        MemoryItem(key="fresh", value="still good"),            # no expiry
        MemoryItem(key="stale", value="old news", expires_at=1.0),  # long past
    ])
    model = SystemCapturingModel()
    agent = Agent(model=model, tools=[search], memory=mem)

    result = await agent.run("q", run_id="r")

    assert "still good" in model.systems[0]
    assert "old news" not in model.systems[0]          # the stale item never reached the prompt
    recalled = [e for e in result.events if e.type == "memory_recalled"][0]
    assert [i["key"] for i in recalled.payload["items"]] == ["fresh"]
    assert "recalled_at" in recalled.payload           # the timestamp is recorded for replay


async def test_recall_failure_is_not_fatal():
    class FailingMemory(Memory):
        async def recall(self, query, *, limit=5):
            raise RuntimeError("vector db down")

        async def remember(self, item):
            return None

    model = SystemCapturingModel()
    agent = Agent(model=model, tools=[search], memory=FailingMemory())

    result = await agent.run("q", run_id="r")

    assert result.status == "completed"                # the run still finishes
    assert result.output == "done"
    recalled = [e for e in result.events if e.type == "memory_recalled"][0]
    assert recalled.payload["items"] == []
    assert "error" in recalled.payload                 # the failure is recorded, not silent
    assert model.systems[0] == ""                      # nothing injected


async def test_null_memory_recalls_nothing_and_leaves_system_unchanged():
    model = SystemCapturingModel()
    agent = Agent(model=model, tools=[search], instructions="base", memory=NullMemory())

    result = await agent.run("q")

    # The recall step still runs (and is recorded), but injects no context.
    assert any(e.type == "memory_recalled" for e in result.events)
    assert model.systems[0] == "base"
