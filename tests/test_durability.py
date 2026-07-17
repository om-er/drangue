"""Phase 2 tests: durable resume, idempotency, and the state-scope seam.

The crash tests simulate a process dying mid-run by making the model raise, then
resuming with a fresh agent against the same store. All offline.
"""

import os
import tempfile

from drangue import (
    Agent,
    InMemoryStore,
    MemoryItem,
    ModelResponse,
    NullMemory,
    SQLiteStore,
    ToolCall,
    tool,
)
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ScriptedModel(Model):
    """Returns scripted responses; the string 'CRASH' raises to fake a crash."""

    def __init__(self, steps):
        self.steps = steps
        self.i = 0
        self.calls = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        step = self.steps[self.i]
        self.i += 1
        if step == "CRASH":
            raise RuntimeError("process died")
        return step


async def test_resume_after_crash_reuses_recorded_steps():
    store = InMemoryStore()
    seen = []

    @tool
    def record(x: int) -> str:
        """Record a number (a side effect)."""
        seen.append(x)
        return "ok"

    # First agent: makes a tool call, records it, then crashes on the next model call.
    crashing = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "record", {"x": 7})]),
        "CRASH",
    ])
    agent_a = Agent(model=crashing, tools=[record], store=store)
    try:
        await agent_a.run("go", run_id="run-1")
        assert False, "expected the run to crash"
    except RuntimeError:
        pass

    assert seen == [7]                 # the side effect happened once, before the crash

    # Second agent: fresh model, same durable store, same run_id -> resume.
    recovering = ScriptedModel([ModelResponse(text="done")])
    agent_b = Agent(model=recovering, tools=[record], store=store)
    result = await agent_b.run("go", run_id="run-1")

    assert result.output == "done"
    assert recovering.calls == 1       # only the remaining model step ran
    assert seen == [7]                 # the tool did NOT run again on resume


async def test_side_effecting_tool_runs_exactly_once_across_resume():
    store = InMemoryStore()
    runs = []

    @tool
    def charge(amount: int, idempotency_key: str = "") -> str:
        """Charge a card once."""
        runs.append(idempotency_key)
        return "charged"

    crashing = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "charge", {"amount": 10})]),
        "CRASH",
    ])
    try:
        await Agent(model=crashing, tools=[charge], store=store).run("pay", run_id="r")
    except RuntimeError:
        pass

    recovering = ScriptedModel([ModelResponse(text="paid")])
    await Agent(model=recovering, tools=[charge], store=store).run("pay", run_id="r")

    assert runs == ["r:2"]             # charged once, with the stable key for seq 2


async def test_resume_of_an_unknown_run_id_raises():
    from drangue import UnknownRunError

    agent = Agent(model=ScriptedModel([ModelResponse(text="hi")]), tools=[add])
    try:
        await agent.resume("typo-run-id")
        assert False, "expected UnknownRunError"
    except UnknownRunError:
        pass
    # Nothing was appended: the typo did not silently create an empty run.
    assert await agent.store.load("typo-run-id") == []


async def test_run_with_a_different_input_on_an_existing_run_is_refused():
    store = InMemoryStore()
    model = ScriptedModel([ModelResponse(text="first answer"),
                           ModelResponse(text="never reached")])
    agent = Agent(model=model, tools=[add], store=store)
    await agent.run("first question", run_id="r1")

    try:
        await agent.run("second question", run_id="r1")
        assert False, "expected ValueError: the new input would be silently dropped"
    except ValueError as exc:
        assert "already exists" in str(exc)

    # Re-running with the SAME input is the replay path and stays legal.
    result = await agent.run("first question", run_id="r1")
    assert result.output == "first answer"
    assert model.calls == 1


async def test_input_guard_still_applies_when_resuming_a_pre_step_crash():
    # A run that crashed after run_started but before any model work must be
    # vetted on resume; the fresh-run branch is not the only door in.
    from drangue import Guardrails
    from drangue.events import Event

    store = InMemoryStore()
    await store.append("r-crashed", Event(seq=0, type="run_started",
                                          payload={"input": "malicious input"}))

    guard = Guardrails(input_guard=lambda text: "looks malicious" if "malicious" in text else None)
    model = ScriptedModel([ModelResponse(text="should never run")])
    agent = Agent(model=model, tools=[add], store=store, guardrails=guard)

    result = await agent.resume("r-crashed")
    assert result.output == "(blocked: looks malicious)"
    assert model.calls == 0


async def test_memory_recall_on_resume_uses_the_recorded_input():
    from drangue.events import Event

    queries = []

    class SpyMemory:
        async def recall(self, query):
            queries.append(query)
            return []

        async def remember(self, item):
            pass

    store = InMemoryStore()
    await store.append("r-mem", Event(seq=0, type="run_started",
                                      payload={"input": "the real question"}))

    model = ScriptedModel([ModelResponse(text="done")])
    agent = Agent(model=model, tools=[add], store=store, memory=SpyMemory())
    await agent.resume("r-mem")

    assert queries == ["the real question"]   # not the empty live input


async def test_idempotency_key_is_stable_and_hidden_from_the_model():
    captured = {}

    @tool
    def act(x: int, idempotency_key: str = "") -> str:
        """Do something idempotently."""
        captured["key"] = idempotency_key
        return "done"

    # The injected parameter is not part of the model-facing schema.
    assert "idempotency_key" not in act.to_schema()["input_schema"]["properties"]
    assert "x" in act.to_schema()["input_schema"]["properties"]

    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        ModelResponse(text="ok"),
    ])
    await Agent(model=model, tools=[act]).run("go", run_id="run-x")

    # seq 0 run_started, seq 1 model_decision, seq 2 tool_result.
    assert captured["key"] == "run-x:2"


async def test_sqlite_store_persists_across_instances():
    path = os.path.join(tempfile.mkdtemp(), "runs.db")
    try:
        store1 = SQLiteStore(path)
        model1 = ScriptedModel([
            ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 2, "b": 3})]),
            ModelResponse(text="5"),
        ])
        first = await Agent(model=model1, tools=[add], store=store1).run("2 + 3", run_id="r")
        assert first.output == "5"
        store1.close()

        # A brand new store instance over the same file: the completed run is
        # replayed from disk without touching the model at all.
        store2 = SQLiteStore(path)
        never = ScriptedModel([])   # would IndexError if generate were called
        second = await Agent(model=never, tools=[add], store=store2).run("2 + 3", run_id="r")
        assert second.output == "5"
        assert never.calls == 0
        store2.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


async def test_null_memory_is_the_default_seam():
    mem = NullMemory()
    assert await mem.recall("anything") == []
    await mem.remember(MemoryItem(key="k", value="v"))   # no-op, does not raise


async def test_model_call_receives_a_stable_idempotency_key():
    class KeyCapturingModel(Model):
        def __init__(self):
            self.keys = []

        async def generate(self, *, system, messages, tools, idempotency_key=None):
            self.keys.append(idempotency_key)
            return ModelResponse(text="done")

    model = KeyCapturingModel()
    await Agent(model=model, tools=[add]).run("go", run_id="rid")
    # The first model decision is seq 1 (seq 0 is run_started), so the request
    # idempotency key is stable across a crash-and-resume.
    assert model.keys == ["rid:1"]
