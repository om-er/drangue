# drangue-postgres

A durable Postgres event store for [drangue](../..), implementing the
`Store` seam. Use it to make runs survive process death across a real database.

```python
from drangue import Agent, SQLiteStore  # swap SQLiteStore for PostgresStore
from drangue_postgres import PostgresStore

agent = Agent("claude-opus-4-8", tools=tools,
              store=PostgresStore("postgresql://localhost/drangue"))
await agent.run("do the thing", run_id="job-42")   # resumable from any process
```

It mirrors the built-in `SQLiteStore` contract: append-only, ordered by seq,
isolated by run_id, payloads round-trip via JSONB, and a duplicate
`(run_id, seq)` is ignored so a retried append cannot duplicate an immutable
event.

## Test

Requires a real Postgres. Set `DRANGUE_POSTGRES_DSN` and the conformance suite
runs; without it, the tests skip.

```bash
export DRANGUE_POSTGRES_DSN=postgresql://localhost/drangue_test
python extensions/run_tests.py        # from the repo root
```

The suite is `drangue.testing.check_store` (+ `check_store_idempotent_append`,
`check_store_with_agent`), the same contract the built-in stores pass.
