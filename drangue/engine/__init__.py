"""The engine seam: the pluggable durability layer.

The default EventSourcedEngine drives the loop with an append-only log. Because
the Agent depends only on this protocol, an adapter can hand orchestration to an
external durable-execution runtime (Temporal, DBOS, ...) without touching agent,
orchestrator, or executor code.
"""

from __future__ import annotations

import typing as t


class Engine(t.Protocol):
    # Takes a RunContext (drangue.context). A single argument keeps this seam
    # stable as capabilities are added as context fields.
    async def run(self, ctx): ...


from .eventsourced import EventSourcedEngine  # noqa: E402

__all__ = ["Engine", "EventSourcedEngine"]
