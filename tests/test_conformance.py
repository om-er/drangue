"""Run the conformance suites against drangue's own built-in implementations.

This proves two things at once: the suites work, and the built-ins satisfy the
seam contracts a battery author will be held to.
"""

import os
import tempfile

from drangue import (
    ConsoleTracer,
    InMemoryStore,
    NullMemory,
    NullTracer,
    RuleRouter,
    SingleModel,
    SQLiteStore,
)
from drangue import EventSourcedEngine
from drangue.testing import (
    FakeModel,
    RecordingTracer,
    check_engine,
    check_memory,
    check_model_interface,
    check_router,
    check_store,
    check_store_idempotent_append,
    check_store_with_agent,
    check_tracer,
)


async def test_in_memory_store_conforms():
    await check_store(InMemoryStore)
    await check_store_with_agent(InMemoryStore)


async def test_sqlite_store_conforms():
    paths = []

    def make():
        path = os.path.join(tempfile.mkdtemp(), "conf.db")
        paths.append(path)
        return SQLiteStore(path)

    try:
        await check_store(make)
        await check_store_idempotent_append(make)   # durable store promises this
        await check_store_with_agent(make)
    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)


async def test_event_sourced_engine_conforms():
    await check_engine(EventSourcedEngine)


async def test_null_memory_conforms():
    await check_memory(NullMemory)


async def test_tracers_conform():
    await check_tracer(NullTracer)
    await check_tracer(RecordingTracer)
    await check_tracer(ConsoleTracer)


async def test_routers_conform():
    await check_router(lambda: SingleModel(FakeModel()))
    await check_router(lambda: RuleRouter(default=FakeModel()))


async def test_fake_model_conforms():
    await check_model_interface(FakeModel(final="x"))


async def test_recording_tracer_captures_spans():
    tracer = RecordingTracer()
    with tracer.span("run", a=1) as span:
        span.set("b", 2)
    name, attrs = tracer.spans[-1]
    assert name == "run"
    assert attrs["a"] == 1 and attrs["b"] == 2
