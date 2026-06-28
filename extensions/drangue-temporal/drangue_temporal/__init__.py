"""drangue-temporal: run drangue agents on Temporal (the Engine, supervised).

Requires the `temporalio` SDK. Importing this package without it raises.
"""

from .engine import (
    REGISTRY,
    AgentRun,
    RunOutcome,
    approve,
    build_worker,
    execute_run,
    record_approval,
    submit_run,
)

__all__ = [
    "REGISTRY",
    "AgentRun",
    "RunOutcome",
    "execute_run",
    "record_approval",
    "build_worker",
    "submit_run",
    "approve",
]
