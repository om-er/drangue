"""In-memory event store. The default, and the dev-time store."""

from __future__ import annotations

from ..events import Event


class InMemoryStore:
    def __init__(self):
        self._logs: dict[str, list[Event]] = {}

    async def append(self, run_id: str, event: Event) -> None:
        self._logs.setdefault(run_id, []).append(event)

    async def load(self, run_id: str) -> list[Event]:
        return list(self._logs.get(run_id, []))
