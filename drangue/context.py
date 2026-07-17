"""RunContext: everything an Engine needs to drive one run.

Capabilities are optional fields, so adding a feature (budget, autonomy, memory,
...) does not widen the `Engine.run` signature. The Agent builds this; an Engine
consumes it. Keeping it a single object is what lets the Engine stay a stable
seam as the runtime grows.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass


@dataclass
class RunContext:
    run_id: str
    input: str
    orchestrator: t.Any
    executor: t.Any
    store: t.Any
    system: str = ""
    tracer: t.Any = None
    emit: t.Any = None          # async callback for streaming events
    budget: t.Any = None
    autonomy: t.Any = None
    memory: t.Any = None
    output_schema: dict | None = None   # constrain the final answer (structured.py)


@dataclass
class RunSpec:
    """A serializable description of a run: which agent and what input.

    This is what crosses a process or worker boundary for a distributed engine.
    The live RunContext (orchestrator, executor with model and tools, guardrails,
    memory) cannot cross a wire; a worker rebuilds it from this spec via an
    AgentRegistry (see drangue.registry.context_from_spec). In-process code never
    needs a RunSpec, it builds a RunContext directly.
    """

    key: str          # registry key naming the agent/toolset to rebuild
    run_id: str
    input: str

    def to_dict(self) -> dict:
        return {"key": self.key, "run_id": self.run_id, "input": self.input}

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        return cls(key=d["key"], run_id=d["run_id"], input=d["input"])
