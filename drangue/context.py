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
