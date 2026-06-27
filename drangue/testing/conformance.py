"""Conformance suites for the drangue seams.

A battery author runs these against their implementation to prove it satisfies
the contract a seam promises. Each check is a plain async function that raises
AssertionError with a clear message on violation and returns None on success, so
it works with any runner (drangue's own `run_tests.py`, pytest, or a script).

Most checks take a zero-argument factory that returns a FRESH instance, so each
assertion starts from a clean slate:

    from drangue.testing import check_store
    await check_store(lambda: MyStore(dsn))

The `Store` contract has a required core (`check_store`) and an opt-in stricter
promise (`check_store_idempotent_append`) that only durable stores need make.
"""

from __future__ import annotations

from ..events import Event
from ..memory import MemoryItem
from .fakes import FakeModel


# --- Store ---------------------------------------------------------------

async def check_store(make_store) -> None:
    """Required Store contract: isolation, ordering, round-trip, empty default."""
    # Unknown run id is empty, never an error.
    s = make_store()
    assert await s.load("unknown") == [], "load(unknown run_id) must return []"

    # Append then load round-trips.
    s = make_store()
    await s.append("r", Event(seq=0, type="run_started", payload={"input": "hi"}))
    loaded = await s.load("r")
    assert len(loaded) == 1, "load must return the appended event"
    assert loaded[0].seq == 0 and loaded[0].type == "run_started", \
        "seq and type must round-trip"

    # Events load in seq order.
    s = make_store()
    for i in range(3):
        await s.append("r", Event(seq=i, type="model_decision", payload={"i": i}))
    assert [e.seq for e in await s.load("r")] == [0, 1, 2], \
        "events must load in seq order"

    # Payload, ts and duration round-trip unchanged (matters for serializing stores).
    s = make_store()
    payload = {"tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1}}],
               "usage": {"input_tokens": 5, "output_tokens": 2}}
    await s.append("r", Event(seq=0, type="model_decision", payload=payload,
                              ts=1.5, duration_ms=2.0))
    e = (await s.load("r"))[0]
    assert e.payload == payload, "payload must round-trip unchanged"
    assert e.ts == 1.5 and e.duration_ms == 2.0, "ts and duration_ms must round-trip"

    # Runs are isolated.
    s = make_store()
    await s.append("a", Event(seq=0, type="x", payload={}))
    assert await s.load("b") == [], "runs must be isolated by run_id"


async def check_store_idempotent_append(make_store) -> None:
    """Opt-in: a durable store should ignore a duplicate (run_id, seq) append.

    The in-memory dev store does not promise this; durable stores should, so a
    retried append after a crash cannot duplicate an immutable event.
    """
    s = make_store()
    await s.append("r", Event(seq=0, type="x", payload={"v": 1}))
    await s.append("r", Event(seq=0, type="x", payload={"v": 2}))   # same seq
    loaded = await s.load("r")
    assert len(loaded) == 1, \
        "a store promising idempotent append must ignore a duplicate (run_id, seq)"


async def check_store_with_agent(make_store) -> None:
    """End-to-end: an Agent backed by this store runs and replays on resume."""
    from ..agent import Agent

    model = FakeModel(final="done")
    agent = Agent(model=model, tools=[], store=make_store())
    result = await agent.run("hi", run_id="smoke")
    assert result.status == "completed" and result.output == "done", \
        "an agent backed by this store must complete a run"

    calls = model.calls
    again = await agent.run("hi", run_id="smoke")
    assert again.output == "done", "a completed run must reload from the store"
    assert model.calls == calls, \
        "a completed run must replay from the store without re-calling the model"


# --- Engine --------------------------------------------------------------

async def check_engine(make_engine) -> None:
    """Engine contract: drive a run to completion and resume by replay.

    Builds a RunContext with an in-memory store and a fake model, so an
    alternative engine (Temporal, DBOS, ...) can verify it honors the core
    promises: complete a simple run, record run_started..run_finished, and on a
    second run with the same id replay from the store without re-calling the model.
    """
    from ..context import RunContext
    from ..executor import Executor
    from ..orchestrator import Orchestrator
    from ..routing import SingleModel
    from ..store import InMemoryStore

    model = FakeModel(final="done")
    store = InMemoryStore()
    executor = Executor(SingleModel(model), {})
    orchestrator = Orchestrator()
    engine = make_engine()

    def ctx(run_id):
        return RunContext(run_id=run_id, input="hi", orchestrator=orchestrator,
                          executor=executor, store=store, system="")

    result = await engine.run(ctx("smoke"))
    assert result.status == "completed", "engine must complete a simple run"
    assert result.output == "done"
    types = [e.type for e in result.events]
    assert types[0] == "run_started" and types[-1] == "run_finished", \
        "engine must record run_started ... run_finished"

    calls = model.calls
    again = await engine.run(ctx("smoke"))
    assert again.output == "done", "a completed run must reload from the store"
    assert model.calls == calls, \
        "a completed run must replay without re-calling the model"


# --- Memory --------------------------------------------------------------

async def check_memory(make_memory) -> None:
    """Memory contract: recall returns a list of items; remember returns None."""
    m = make_memory()
    items = await m.recall("anything")
    assert isinstance(items, list), "recall must return a list"
    for it in items:
        assert hasattr(it, "value"), "recalled items must be MemoryItem-like (.value)"
    result = await m.remember(MemoryItem(key="k", value="v"))
    assert result is None, "remember must return None"


# --- Tracer --------------------------------------------------------------

async def check_tracer(make_tracer) -> None:
    """Tracer contract: span() is a context manager with set(); spans nest."""
    tracer = make_tracer()
    with tracer.span("run", run_id="r") as span:
        span.set("text", "hello")
        span.set("count", 1)
        span.set("obj", {"nested": [1, 2]})
        with tracer.span("child", seq=0) as child:
            child.set("flag", True)
    # Reaching here without raising is the contract.


# --- Router --------------------------------------------------------------

async def check_router(make_router) -> None:
    """Router contract: choose returns a model and is deterministic per input."""
    router = make_router()
    first = router.choose(messages=[], step_index=0)
    assert first is not None, "choose must return a model"
    second = router.choose(messages=[], step_index=0)
    assert first is second, "choose must be deterministic for identical inputs"


# --- Model ---------------------------------------------------------------

async def check_model_interface(model) -> None:
    """Model contract: generate accepts the standard kwargs and returns a
    ModelResponse with the expected fields. Pass a model wired to a stub or
    offline client so this runs without a network call."""
    resp = await model.generate(
        system="", messages=[{"role": "user", "content": "hi"}],
        tools=[], idempotency_key=None,
    )
    assert isinstance(resp.text, str), "ModelResponse.text must be a str"
    assert isinstance(resp.tool_calls, list), "ModelResponse.tool_calls must be a list"
    assert hasattr(resp, "usage"), "ModelResponse must expose usage"
    assert hasattr(resp, "reasoning"), "ModelResponse must expose reasoning"
