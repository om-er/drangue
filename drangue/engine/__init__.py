"""The engine seam: the pluggable durability layer.

The default EventSourcedEngine drives the loop with an append-only log. Because
the Agent depends only on this protocol, an adapter can hand orchestration to an
external durable-execution runtime (Temporal, DBOS, ...) without touching agent,
orchestrator, or executor code.
"""

from __future__ import annotations

import typing as t


class Engine(t.Protocol):
    async def run(self, *, run_id, orchestrator, executor, store, system, input, emit=None): ...


from .eventsourced import EventSourcedEngine  # noqa: E402

__all__ = ["Engine", "EventSourcedEngine"]
