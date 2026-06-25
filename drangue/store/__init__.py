"""The store seam: where the event log lives.

Phase 0 ships InMemoryStore. Phase 2 adds a durable SQLiteStore behind the same
protocol, at which point a run survives process death with no code change above
this line.
"""

from __future__ import annotations

import typing as t

from ..events import Event


class Store(t.Protocol):
    async def append(self, run_id: str, event: Event) -> None: ...
    async def load(self, run_id: str) -> list[Event]: ...


from .memory import InMemoryStore  # noqa: E402

__all__ = ["Store", "InMemoryStore"]
