"""drangue-postgres conformance.

Requires a real Postgres. Set DRANGUE_POSTGRES_DSN (e.g.
postgresql://localhost/drangue_test) to run; otherwise these skip. Each factory
call uses a fresh table so the conformance suite's clean-store assumption holds.
"""

import itertools
import os

from drangue.testing import (
    check_store,
    check_store_idempotent_append,
    check_store_with_agent,
)

DSN = os.environ.get("DRANGUE_POSTGRES_DSN")
_counter = itertools.count()


def _make():
    from drangue_postgres import PostgresStore
    return PostgresStore(DSN, table=f"conf_events_{next(_counter)}")


async def test_postgres_store_conforms():
    if not DSN:
        print("SKIP test_postgres_store_conforms: set DRANGUE_POSTGRES_DSN to run")
        return
    await check_store(_make)
    await check_store_idempotent_append(_make)   # durable store promises this
    await check_store_with_agent(_make)
