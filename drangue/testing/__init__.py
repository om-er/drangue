"""drangue.testing: conformance suites and offline fakes for battery authors.

    from drangue.testing import check_store, FakeModel

    async def test_my_store_conforms():
        await check_store(lambda: MyStore(dsn))
        await check_store_idempotent_append(lambda: MyStore(dsn))
        await check_store_with_agent(lambda: MyStore(dsn))

This module is import-light and has no test-framework dependency, so the checks
run under pytest, drangue's own runner, or a plain script.
"""

from .conformance import (
    check_memory,
    check_model_interface,
    check_router,
    check_store,
    check_store_idempotent_append,
    check_store_with_agent,
    check_tracer,
)
from .fakes import FakeModel, RecordingTracer

__all__ = [
    "FakeModel",
    "RecordingTracer",
    "check_store",
    "check_store_idempotent_append",
    "check_store_with_agent",
    "check_memory",
    "check_tracer",
    "check_router",
    "check_model_interface",
]
