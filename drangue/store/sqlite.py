"""A durable, append-only event store backed by SQLite.

This is what turns the in-process replay of Phase 0/1 into real crash recovery:
the log outlives the process. Point an Agent at one and a run can be resumed in a
brand new process by passing its run_id.

The primary key (run_id, seq) plus INSERT OR IGNORE makes appends idempotent:
the event at a given seq is immutable, so a retried append after a crash between
"append" and "continue" is a no-op rather than a duplicate. SQLite calls are
blocking, so they run off the event loop via asyncio.to_thread, serialized by a
lock.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from ..events import Event


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "  run_id TEXT, seq INTEGER, type TEXT, payload TEXT,"
            "  ts REAL, duration_ms REAL,"
            "  PRIMARY KEY (run_id, seq)"
            ")"
        )
        self._conn.commit()

    async def append(self, run_id: str, event: Event) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append, run_id, event)

    def _append(self, run_id: str, event: Event) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO events "
            "(run_id, seq, type, payload, ts, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, event.seq, event.type, json.dumps(event.payload),
             event.ts, event.duration_ms),
        )
        self._conn.commit()

    async def load(self, run_id: str) -> list[Event]:
        async with self._lock:
            rows = await asyncio.to_thread(self._load, run_id)
        return [
            Event(seq=r[0], type=r[1], payload=json.loads(r[2]),
                  ts=r[3], duration_ms=r[4])
            for r in rows
        ]

    def _load(self, run_id: str):
        cur = self._conn.execute(
            "SELECT seq, type, payload, ts, duration_ms FROM events "
            "WHERE run_id = ? ORDER BY seq",
            (run_id,),
        )
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
