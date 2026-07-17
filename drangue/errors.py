"""Tool failure categories (Chapter 6).

A tool, or the integration wrapper inside it, raises one of these to tell the
runtime what kind of failure happened. The runtime uses the category to decide
whether to retry and how to report a clean failure to the model. Transient
failures are retried; permanent and auth failures are not (auth needs a refresh,
not a blind retry); validation failures mean the upstream contract drifted.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for classified tool failures."""


class TransientError(ToolError):
    """A temporary failure worth retrying (a blip, a brief outage)."""


class PermanentError(ToolError):
    """A failure that will not improve on retry."""


class AuthError(ToolError):
    """Credentials were rejected. Needs a refresh, not a blind retry."""


class RateLimitError(TransientError):
    """Throttled by the upstream. Carries an optional Retry-After hint."""

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(ToolError):
    """The response did not match the expected contract (schema drift)."""


class UnknownRunError(KeyError):
    """A run_id that does not exist in the store was asked to resume."""
