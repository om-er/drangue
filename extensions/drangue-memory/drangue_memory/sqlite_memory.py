"""SQLite-backed embedding memory: a drop-in drangue Memory battery.

Implements the `Memory` seam (recall/remember). The embedder is a deterministic
local hash embedder so the package runs offline with no vector DB and no model,
which is what lets its conformance run in CI. For production, swap the backend
(pgvector, etc.) and the embedder (a real embedding model); the recall/remember
contract is unchanged, so `drangue.testing.check_memory` still applies.

Staleness is left to the runtime: recall returns items with their `expires_at`,
and the engine drops expired ones at recall time against a recorded timestamp
(see drangue.memory). So recall here stays a pure function of the DB, with no
clock read, which keeps replay deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3

from drangue.memory import MemoryItem

_DIM = 64


def _embed(text: str) -> list[float]:
    """Deterministic bag-of-words hash embedding (md5, not the randomized hash)."""
    vec = [0.0] * _DIM
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SqliteMemory:
    """Embedding memory over SQLite. One instance per process."""

    def __init__(self, path: str, *, embed=_embed):
        self.path = path
        self._embed = embed
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "  key TEXT PRIMARY KEY, value TEXT, expires_at REAL, embedding TEXT"
            ")"
        )
        self._conn.commit()

    async def remember(self, item: MemoryItem) -> None:
        embedding = json.dumps(self._embed(item.value))
        async with self._lock:
            await asyncio.to_thread(self._remember, item, embedding)

    def _remember(self, item: MemoryItem, embedding: str) -> None:
        self._conn.execute(
            "INSERT INTO memories (key, value, expires_at, embedding) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "expires_at=excluded.expires_at, embedding=excluded.embedding",
            (item.key, item.value, item.expires_at, embedding),
        )
        self._conn.commit()

    async def recall(self, query: str, *, limit: int = 5) -> list[MemoryItem]:
        q = self._embed(query)
        async with self._lock:
            rows = await asyncio.to_thread(self._all)
        scored = []
        for key, value, expires_at, embedding in rows:
            score = _cosine(q, json.loads(embedding))
            if score > 0:
                scored.append((score, key, value, expires_at))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [MemoryItem(key=k, value=v, expires_at=e)
                for _score, k, v, e in scored[:limit]]

    def _all(self):
        cur = self._conn.execute(
            "SELECT key, value, expires_at, embedding FROM memories")
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
