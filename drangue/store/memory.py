"""In-memory event store. The default, and the dev-time store.

Same append semantics as the durable stores (identical retry is a no-op, a
conflicting event at an occupied seq raises ConflictError), so behavior does
not quietly change when a durable store is swapped in. Payloads are deep-copied
on the way in and out: the log is the source of truth, and a caller mutating a
loaded event must not be able to rewrite history.
"""

from __future__ import annotations

import copy

from ..errors import ConflictError
from ..events import Event


def _copy(event: Event) -> Event:
    return Event(seq=event.seq, type=event.type,
                 payload=copy.deepcopy(event.payload),
                 ts=event.ts, duration_ms=event.duration_ms)


class InMemoryStore:
    def __init__(self):
        self._logs: dict[str, list[Event]] = {}

    async def append(self, run_id: str, event: Event) -> None:
        log = self._logs.setdefault(run_id, [])
        existing = next((e for e in log if e.seq == event.seq), None)
        if existing is not None:
            if existing.type == event.type and existing.payload == event.payload:
                return  # a retried append of the same immutable event
            raise ConflictError(run_id, event.seq)
        log.append(_copy(event))

    async def load(self, run_id: str) -> list[Event]:
        return [_copy(e) for e in self._logs.get(run_id, [])]
