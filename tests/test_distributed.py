"""The distributed construction path: a RunSpec crosses a (simulated) worker
boundary and the agent is rebuilt from a registry. This is what a Temporal/DBOS
engine needs; here it runs offline by JSON round-tripping the spec and rebuilding
on the "worker" side, with a shared SQLite store standing in for a shared DB.
"""

import json
import os
import tempfile

from drangue import (
    Agent,
    AgentRegistry,
    EventSourcedEngine,
    RunSpec,
    SQLiteStore,
    context_from_spec,
)
from drangue.testing import FakeModel


async def test_spec_crosses_boundary_rebuilds_and_runs():
    path = os.path.join(tempfile.mkdtemp(), "dist.db")
    models = []

    def build_greeter():
        model = FakeModel(final="hello from the worker")
        models.append(model)
        # The worker rebuilds the agent pointing at the shared durable store.
        return Agent(model=model, tools=[], store=SQLiteStore(path))

    registry = AgentRegistry()
    registry.register("greeter", build_greeter)

    try:
        # Client side: a serializable spec crosses the wire as JSON.
        wire = json.dumps(RunSpec("greeter", "job-1", "hi").to_dict())

        # Worker side: deserialize, rebuild the live context, run.
        spec = RunSpec.from_dict(json.loads(wire))
        ctx = context_from_spec(spec, registry)
        result = await EventSourcedEngine().run(ctx)

        assert result.status == "completed"
        assert result.output == "hello from the worker"
        assert models[0].calls == 1

        # A second worker invocation (retry/restart) resumes from the shared
        # store and replays without re-calling the model.
        spec2 = RunSpec.from_dict(json.loads(wire))
        ctx2 = context_from_spec(spec2, registry)
        result2 = await EventSourcedEngine().run(ctx2)

        assert result2.output == "hello from the worker"
        assert models[1].calls == 0     # replayed across the boundary, not re-run
    finally:
        if os.path.exists(path):
            os.remove(path)


async def test_unknown_key_is_a_clear_error():
    registry = AgentRegistry()
    spec = RunSpec("missing", "r", "hi")
    try:
        context_from_spec(spec, registry)
        assert False, "expected KeyError for an unregistered key"
    except KeyError as exc:
        assert "missing" in str(exc)
