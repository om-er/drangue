# drangue-memory

Long-term memory for [drangue](../..), implementing the `Memory` seam.

`SqliteMemory` is an embedding memory over SQLite with a deterministic local
embedder, so it runs offline with no vector DB or model. Drop it into an agent:

```python
from drangue import Agent, MemoryItem
from drangue_memory import SqliteMemory

mem = SqliteMemory("memory.db")
await mem.remember(MemoryItem(key="inc-1", value="payments latency: downstream timeout"))

agent = Agent("claude-opus-4-8", tools=tools, memory=mem)
# the engine recalls against the run input and injects it into the model's context
await agent.run("investigate payments latency", run_id="inc-2")
```

Recall returns items with their `expires_at`; the runtime drops expired ones at
recall time (see drangue's memory wiring), so recall here stays a pure function
of the DB.

For production, swap the backend (pgvector) and the embedder (a real model); the
`recall`/`remember` contract is unchanged, so `drangue.testing.check_memory`
still applies.

## Test

From the repo root, the shared runner runs every extension's tests:

```bash
python extensions/run_tests.py
```
