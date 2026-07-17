"""Postgres-backed event store: a durable drangue Store battery.

Mirrors the built-in SQLiteStore's contract (drangue.testing.check_store):
append-only, ordered by seq, isolated by run_id, payload round-trips, a retried
append of the same event is a no-op, and a DIFFERENT event at an occupied
(run_id, seq) raises drangue.ConflictError instead of silently dropping either
side of the race. Uses async psycopg; one instance per process, connecting
lazily.

The `table` parameter exists so conformance tests can isolate runs in their own
table; in normal use the default is fine.
"""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from drangue.errors import ConflictError
from drangue.events import Event

_IDENT_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class PostgresStore:
    def __init__(self, dsn: str, *, table: str = "events"):
        if not table or any(c not in _IDENT_OK for c in table):
            raise ValueError(f"invalid table name: {table!r}")
        self.dsn = dsn
        self.table = table
        self._conn = None

    async def _connection(self):
        if self._conn is None:
            self._conn = await psycopg.AsyncConnection.connect(self.dsn, autocommit=True)
            await self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "  run_id TEXT, seq INTEGER, type TEXT, payload JSONB,"
                "  ts DOUBLE PRECISION, duration_ms DOUBLE PRECISION,"
                "  PRIMARY KEY (run_id, seq))"
            )
        return self._conn

    async def append(self, run_id: str, event: Event) -> None:
        conn = await self._connection()
        try:
            await conn.execute(
                f"INSERT INTO {self.table} "
                "(run_id, seq, type, payload, ts, duration_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (run_id, event.seq, event.type, Jsonb(event.payload),
                 event.ts, event.duration_ms),
            )
        except psycopg.errors.UniqueViolation:
            cur = await conn.execute(
                f"SELECT type, payload FROM {self.table} "
                "WHERE run_id = %s AND seq = %s",
                (run_id, event.seq),
            )
            row = await cur.fetchone()
            if row is not None and row[0] == event.type and row[1] == event.payload:
                return  # a retried append of the same immutable event
            raise ConflictError(run_id, event.seq) from None

    async def load(self, run_id: str) -> list[Event]:
        conn = await self._connection()
        cur = await conn.execute(
            f"SELECT seq, type, payload, ts, duration_ms FROM {self.table} "
            "WHERE run_id = %s ORDER BY seq",
            (run_id,),
        )
        rows = await cur.fetchall()
        # psycopg returns JSONB as a Python object already.
        return [Event(seq=r[0], type=r[1], payload=r[2], ts=r[3], duration_ms=r[4])
                for r in rows]

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
