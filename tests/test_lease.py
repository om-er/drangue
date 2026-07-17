"""Per-run leases: one live driver per run, crash recovery unaffected.

The store-level contract is covered by drangue.testing.check_store_lease;
these exercise the engine behavior around it. All offline.
"""

import asyncio

from drangue import Agent, InMemoryStore, LeaseHeldError, ModelResponse, ToolCall, tool
from drangue.models import Model


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        step = self.steps[self.i]
        self.i += 1
        if step == "CRASH":
            raise RuntimeError("process died")
        return step


async def test_a_second_live_driver_is_refused_while_the_first_runs():
    store = InMemoryStore()
    gate = asyncio.Event()
    entered = asyncio.Event()

    @tool
    async def slow() -> str:
        """Holds the run open."""
        entered.set()
        await asyncio.wait_for(gate.wait(), timeout=5)
        return "done"

    model_a = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "slow", {})]),
        ModelResponse(text="finished"),
    ])
    agent_a = Agent(model=model_a, tools=[slow], store=store)
    driving = asyncio.create_task(agent_a.run("go", run_id="one"))
    await entered.wait()                     # agent A is live inside the tool

    agent_b = Agent(model=ScriptedModel([ModelResponse(text="hijack")]),
                    tools=[slow], store=store)
    try:
        await agent_b.run("go", run_id="one")
        assert False, "expected LeaseHeldError while another driver is live"
    except LeaseHeldError:
        pass

    gate.set()
    result = await driving
    assert result.output == "finished"

    # After completion the lease is released; a replay works immediately.
    replay = await agent_b.run("go", run_id="one")
    assert replay.output == "finished"


async def test_a_crashed_driver_releases_and_the_run_resumes():
    store = InMemoryStore()
    crashing = ScriptedModel(["CRASH"])
    try:
        await Agent(model=crashing, tools=[], store=store).run("go", run_id="c")
    except RuntimeError:
        pass

    recovering = ScriptedModel([ModelResponse(text="recovered")])
    result = await Agent(model=recovering, tools=[], store=store).run("go", run_id="c")
    assert result.output == "recovered"


async def test_a_dead_owners_lease_lapses_on_ttl():
    store = InMemoryStore()
    # A process died without releasing; its lease had a tiny TTL.
    assert await store.acquire_lease("orphan", "dead-process", 0.0)

    model = ScriptedModel([ModelResponse(text="took over")])
    result = await Agent(model=model, tools=[], store=store).run("go", run_id="orphan")
    assert result.output == "took over"


async def test_a_live_foreign_lease_blocks_the_engine():
    store = InMemoryStore()
    assert await store.acquire_lease("held", "someone-else", 60.0)

    agent = Agent(model=ScriptedModel([ModelResponse(text="x")]), tools=[],
                  store=store)
    try:
        await agent.run("go", run_id="held")
        assert False, "expected LeaseHeldError"
    except LeaseHeldError as exc:
        assert exc.run_id == "held"
