"""Multi-turn conversations: follow-up inputs into a completed run.

A follow-up is a user_message event in the same log, so the whole
conversation replays, resumes, and folds like any other run. All offline.
"""

import json

from drangue import Agent, Guardrails, InMemoryStore, ModelResponse, ToolCall, tool
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.seen_messages = []

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.seen_messages.append([dict(m) for m in messages])
        step = self.steps[self.i]
        self.i += 1
        if step == "CRASH":
            raise RuntimeError("process died")
        return step


async def test_follow_up_turn_sees_the_whole_conversation():
    model = ScriptedModel([
        ModelResponse(text="2+2 is 4"),
        ModelResponse(text="3+3 is 6"),
    ])
    agent = Agent(model=model, tools=[add])

    first = await agent.run("what is 2+2?", run_id="chat")
    assert first.output == "2+2 is 4"

    second = await agent.run("and 3+3?", run_id="chat")
    assert second.output == "3+3 is 6"

    # The second model call saw the full prior conversation.
    turn2 = model.seen_messages[1]
    contents = [m.get("content") for m in turn2]
    assert contents == ["what is 2+2?", "2+2 is 4", "and 3+3?"]

    # Each turn ends with its own recorded run_finished.
    finishes = [e for e in second.events if e.type == "run_finished"]
    assert [f.payload["output"] for f in finishes] == ["2+2 is 4", "3+3 is 6"]
    assert second.status == "completed"


async def test_follow_up_can_use_tools():
    model = ScriptedModel([
        ModelResponse(text="hello"),
        ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 5, "b": 6})]),
        ModelResponse(text="it is 11"),
    ])
    agent = Agent(model=model, tools=[add])
    await agent.run("hi", run_id="chat2")
    result = await agent.run("what is 5+6?", run_id="chat2")

    assert result.output == "it is 11"
    tool_results = [e for e in result.events if e.type == "tool_result"]
    assert tool_results[0].payload["content"] == "11"


async def test_a_run_in_flight_still_refuses_a_new_input():
    from drangue.events import Event

    store = InMemoryStore()
    # A crashed run: started, never finished.
    await store.append("stuck", Event(seq=0, type="run_started",
                                      payload={"input": "original"}))
    agent = Agent(model=ScriptedModel([ModelResponse(text="x")]), tools=[],
                  store=store)
    try:
        await agent.run("something else", run_id="stuck")
        assert False, "expected ValueError for a new input on an in-flight run"
    except ValueError as exc:
        assert "in flight" in str(exc)


async def test_crash_retry_of_a_follow_up_does_not_duplicate_the_turn():
    store = InMemoryStore()
    crashing = ScriptedModel([ModelResponse(text="turn one"), "CRASH"])
    agent_a = Agent(model=crashing, tools=[], store=store)
    await agent_a.run("first", run_id="c")
    try:
        await agent_a.run("second", run_id="c")   # user_message lands, model dies
        assert False, "expected the crash"
    except RuntimeError:
        pass

    recovering = ScriptedModel([ModelResponse(text="turn two")])
    agent_b = Agent(model=recovering, tools=[], store=store)
    result = await agent_b.run("second", run_id="c")   # same follow-up retried

    assert result.output == "turn two"
    user_messages = [e for e in result.events if e.type == "user_message"]
    assert len(user_messages) == 1        # not duplicated by the retry


async def test_follow_up_input_is_vetted_by_the_input_guard():
    guard = Guardrails(
        input_guard=lambda text: "nope" if "malicious" in text else None)
    model = ScriptedModel([ModelResponse(text="clean answer")])
    agent = Agent(model=model, tools=[], guardrails=guard)

    await agent.run("clean question", run_id="g")
    blocked = await agent.run("malicious follow-up", run_id="g")

    assert blocked.output == "(blocked: nope)"
    assert model.i == 1                   # the model never saw turn two


async def test_replaying_the_original_input_stays_a_replay():
    model = ScriptedModel([ModelResponse(text="four")])
    agent = Agent(model=model, tools=[])
    await agent.run("2+2?", run_id="r")
    again = await agent.run("2+2?", run_id="r")   # same input: replay, not a turn

    assert again.output == "four"
    assert model.i == 1
    assert not [e for e in again.events if e.type == "user_message"]
