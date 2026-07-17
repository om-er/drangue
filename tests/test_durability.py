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


async def test_breaking_out_of_stream_does_not_lose_an_executed_step():
    import asyncio

    executed = []

    @tool
    def act(x: int) -> str:
        """A side effect."""
        executed.append(x)
        return "acted"

    class SlowAppendStore(InMemoryStore):
        """Yields control inside append, so a pending cancellation (from the
        consumer's break) is delivered mid-write — the crash window."""

        async def append(self, run_id, event):
            await asyncio.sleep(0)
            await super().append(run_id, event)

    store = SlowAppendStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[act], store=store)

    stream = agent.stream("go", run_id="r-stream")
    async for ev in stream:
        if ev.type == "tool_result":
            break                       # consumer walks away mid-run
    # Closing the stream cancels the engine task deterministically (a bare
    # break defers generator cleanup to GC). The cancellation lands mid-step;
    # the shielded append must still record whatever already executed.
    await stream.aclose()

    assert executed == [1]
    events = await store.load("r-stream")
    recorded = [e for e in events if e.type == "tool_result"]
    assert len(recorded) == 1           # the executed step IS in the log

    # Resuming completes without re-running the side effect.
    result = await agent.resume("r-stream")
    assert result.output == "done"
    assert executed == [1]


async def test_irreversible_tool_is_not_reexecuted_after_a_record_crash():
    import json

    executed = []

    @tool(reversible=False)
    def wire_funds(amount: int) -> str:
        """Wire money (irreversible)."""
        executed.append(amount)
        return "wired"

    class CrashAfterExecuteStore(InMemoryStore):
        """Drops the tool_result append once, simulating a crash after the
        side effect ran but before its record landed."""

        def __init__(self):
            super().__init__()
            self.crashed = False

        async def append(self, run_id, event):
            if not self.crashed and event.type == "tool_result":
                self.crashed = True
                raise RuntimeError("process died before the record landed")
            await super().append(run_id, event)

    store = CrashAfterExecuteStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "wire_funds", {"amount": 100})]),
    ])
    agent = Agent(model=model, tools=[wire_funds], store=store)
    try:
        await agent.run("pay", run_id="r-wire")
    except RuntimeError:
        pass
    assert executed == [100]            # the effect happened, unrecorded

    # A new process resumes: the orphaned step_started intent means the
    # outcome is unknown — the tool must NOT run again.
    recovering = ScriptedModel([ModelResponse(text="escalating to a human")])
    result = await Agent(model=recovering, tools=[wire_funds],
                         store=store).run("pay", run_id="r-wire")

    assert executed == [100]            # NOT re-executed
    tool_results = [e for e in result.events if e.type == "tool_result"]
    parsed = json.loads(tool_results[0].payload["content"])
    assert parsed["error"]["category"] == "unknown_outcome"
    assert result.output == "escalating to a human"


async def test_reversible_tools_stay_at_least_once_with_no_intent_overhead():
    store = InMemoryStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 1, "b": 1})]),
        ModelResponse(text="done"),
    ])
    result = await Agent(model=model, tools=[add], store=store).run("go", run_id="r-rev")
    assert result.output == "done"
    assert not [e for e in result.events if e.type == "step_started"]


class RacingStore(InMemoryStore):
    """Injects a competitor's event at a chosen seq just before our append,
    simulating a second process resuming the same run concurrently."""

    def __init__(self, race_at_type, competitor_payload):
        super().__init__()
        self.race_at_type = race_at_type
        self.competitor_payload = competitor_payload
        self.raced = False

    async def append(self, run_id, event):
        if not self.raced and event.type == self.race_at_type:
            self.raced = True
            from drangue.events import Event as Ev
            await super().append(run_id, Ev(seq=event.seq, type=event.type,
                                            payload=self.competitor_payload))
        await super().append(run_id, event)


async def test_a_lost_write_race_folds_the_winner_instead_of_crashing():
    # Two engines on one run_id: the loser's tool_result must be discarded in
    # favor of the recorded winner, and the run must complete from the
    # winner's log — not crash, not overwrite.
    store = RacingStore(
        race_at_type="tool_result",
        competitor_payload={"call_id": "c1", "name": "add", "content": "9"},
    )
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 2, "b": 3})]),
        ModelResponse(text="done"),
    ])
    result = await Agent(model=model, tools=[add], store=store).run("go", run_id="r-race")

    assert result.output == "done"
    tool_results = [e for e in result.events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].payload["content"] == "9"   # the winner's fact, kept
    # And the conversation the model saw was folded from the winner's log.
    tool_msg = [m for m in result.messages if m.get("role") == "tool"][0]
    assert tool_msg["content"] == "9"


async def test_approval_survives_a_concurrent_seq_claim():
    from drangue import Autonomy

    ran = []

    @tool
    def act(x: int) -> str:
        """Act."""
        ran.append(x)
        return "acted"

    store = InMemoryStore()
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "act", {"x": 1})]),
        ModelResponse(text="done"),
    ])
    agent = Agent(model=model, tools=[act], store=store,
                  autonomy=Autonomy(modes={"act": "assisted"}))
    paused = await agent.run("go", run_id="r-appr")
    assert paused.status == "paused"

    # A competitor claims the seq approve() would use; approve must retry at
    # the new tail rather than losing the human's decision.
    from drangue import rollout as _r
    from drangue.events import Event as Ev
    events = await store.load("r-appr")
    await store.append("r-appr", Ev(seq=_r.next_seq(events), type="memory_recalled",
                                    payload={"items": [], "recalled_at": 0.0}))

    await agent.approve("r-appr")
    result = await agent.resume("r-appr")
    assert result.output == "done"
    assert ran == [1]


async def test_a_crashed_step_is_recorded_as_run_failed():
    from drangue.events import Result

    store = InMemoryStore()

    @tool
    def noop() -> str:
        """No-op."""
        return "x"

    model = ScriptedModel(["CRASH"])
    agent = Agent(model=model, tools=[noop], store=store, model_retries=0)
    try:
        await agent.run("go", run_id="r-fail")
        assert False, "expected the crash to propagate"
    except RuntimeError:
        pass

    events = await store.load("r-fail")
    assert events[-1].type == "run_failed"
    assert events[-1].payload["error"] == "process died"
    # The persisted run reads as failed, not as running forever.
    result = Result(output="", events=events)
    assert result.status == "failed"
    assert result.error == "process died"

    # Resume retries from the failed step and completes.
    recovered = await Agent(model=ScriptedModel([ModelResponse(text="done")]),
                            tools=[noop], store=store).run("go", run_id="r-fail")
    assert recovered.output == "done"
    assert recovered.status == "completed"


async def test_transient_model_errors_are_retried():
    from drangue import TransientError

    class FlakyOnce(Model):
        def __init__(self):
            self.calls = 0

        async def generate(self, *, system, messages, tools, idempotency_key=None):
            self.calls += 1
            if self.calls == 1:
                raise TransientError("blip")
            return ModelResponse(text="recovered")

    model = FlakyOnce()
    result = await Agent(model=model, tools=[]).run("go")
    assert result.output == "recovered"
    assert model.calls == 2
    assert result.status == "completed"      # no run_failed recorded


async def test_truncated_output_is_flagged():
    model = ScriptedModel([ModelResponse(text="cut of", stop_reason="max_tokens")])
    result = await Agent(model=model, tools=[]).run("go")
    assert result.status == "completed"
    assert result.truncated is True

    model = ScriptedModel([ModelResponse(text="whole", stop_reason="end_turn")])
    result = await Agent(model=model, tools=[]).run("go")
    assert result.truncated is False


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


async def test_same_input_on_an_existing_run_replays_without_a_model_call():
    # A new input on a COMPLETED run opens a follow-up turn (see
    # tests/test_multiturn.py); the same input stays a pure replay, and an
    # in-flight run still refuses a different input.
    store = InMemoryStore()
    model = ScriptedModel([ModelResponse(text="first answer")])
    agent = Agent(model=model, tools=[add], store=store)
    await agent.run("first question", run_id="r1")

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
