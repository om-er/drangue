"""A durable, append-only event store backed by SQLite.

This is what turns the in-process replay of Phase 0/1 into real crash recovery:
the log outlives the process. Point an Agent at one and a run can be resumed in a
brand new process by passing its run_id.

The primary key (run_id, seq) makes the event at a given seq immutable. A
retried append of the SAME event (a crash between "append" and "continue") is
an idempotent no-op; a DIFFERENT event at an occupied seq raises ConflictError,
because silently dropping either side of that race loses an approval or
re-executes a side effect.

Concurrency: WAL mode plus a busy timeout let a second process (say, an
approval CLI) write while a run is in flight, instead of dying on "database is
locked". In-process, the single shared connection is serialized by a
threading.Lock — an asyncio.Lock would only serialize within one event loop,
and SQLite calls run in worker threads via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

from ..errors import ConflictError
from ..events import Event


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "  run_id TEXT, seq INTEGER, type TEXT, payload TEXT,"
            "  ts REAL, duration_ms REAL,"
            "  PRIMARY KEY (run_id, seq)"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS leases ("
            "  run_id TEXT PRIMARY KEY, owner TEXT, expires_at REAL"
            ")"
        )
        self._conn.commit()

    async def append(self, run_id: str, event: Event) -> None:
        await asyncio.to_thread(self._append, run_id, event)

    def _append(self, run_id: str, event: Event) -> None:
        payload = json.dumps(event.payload)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO events "
                    "(run_id, seq, type, payload, ts, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, event.seq, event.type, payload,
                     event.ts, event.duration_ms),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT type, payload FROM events WHERE run_id = ? AND seq = ?",
                    (run_id, event.seq),
                ).fetchone()
                if row is not None and row[0] == event.type and row[1] == payload:
                    return  # a retried append of the same immutable event
                raise ConflictError(run_id, event.seq) from None

    async def load(self, run_id: str) -> list[Event]:
        rows = await asyncio.to_thread(self._load, run_id)
        return [
            Event(seq=r[0], type=r[1], payload=json.loads(r[2]),
                  ts=r[3], duration_ms=r[4])
            for r in rows
        ]

    def _load(self, run_id: str):
        with self._lock:
            cur = self._conn.execute(
                "SELECT seq, type, payload, ts, duration_ms FROM events "
                "WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
            return cur.fetchall()

    # Per-run lease: one live driver per run, across processes. Reentrant for
    # the same owner (re-acquiring is how the engine renews); a dead owner's
    # lease lapses on its TTL, so crash recovery needs no cleanup step.
    async def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        return await asyncio.to_thread(self._acquire_lease, run_id, owner, ttl_s)

    def _acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        with self._lock:
            now = time.time()
            row = self._conn.execute(
                "SELECT owner, expires_at FROM leases WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is not None and row[0] != owner and row[1] > now:
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO leases (run_id, owner, expires_at) "
                "VALUES (?, ?, ?)",
                (run_id, owner, now + ttl_s),
            )
            self._conn.commit()
            return True

    async def release_lease(self, run_id: str, owner: str) -> None:
        await asyncio.to_thread(self._release_lease, run_id, owner)

    def _release_lease(self, run_id: str, owner: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM leases WHERE run_id = ? AND owner = ?",
                (run_id, owner),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
