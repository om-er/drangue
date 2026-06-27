"""drangue: a tiny, obvious agent runtime for Python.

The simple path is one object and one call. Underneath, runs are driven as an
event-sourced loop (orchestrator decides, executor acts, store records) so the
same facade grows into durability, observability, and recovery without changing.
"""

from .agent import Agent
from .budget import Budget
from .engine import Engine, EventSourcedEngine
from .errors import (
    AuthError,
    PermanentError,
    RateLimitError,
    ToolError,
    TransientError,
    ValidationError,
)
from .evals import (
    Check,
    EvalReport,
    Gate,
    GateDecision,
    Judge,
    Scenario,
    ScenarioResult,
    evaluate,
    forbids_tool,
    judged,
    output_contains,
    output_satisfies,
    scenario_from_result,
    tools_called,
    within_steps,
    within_tokens,
)
from .events import Event, Result, Span
from .guardrails import Guardrails
from .hardening import ToolPolicy
from .memory import Memory, MemoryItem, NullMemory
from .models import AnthropicModel, Model, ModelResponse, OpenAIModel, ToolCall
from .context import RunContext, RunSpec
from .observability import ConsoleTracer, NullTracer, OTelTracer, Tracer
from .registry import AgentRegistry, context_from_spec
from .rollout import Autonomy
from .routing import RuleRouter, Router, SingleModel
from .store import InMemoryStore, SQLiteStore, Store
from .tool import Tool, harden, tool

__all__ = [
    "Agent",
    "tool",
    "Tool",
    "harden",
    "ToolPolicy",
    "ToolError",
    "TransientError",
    "PermanentError",
    "AuthError",
    "RateLimitError",
    "ValidationError",
    "Event",
    "Result",
    "Span",
    "Model",
    "AnthropicModel",
    "OpenAIModel",
    "ModelResponse",
    "ToolCall",
    "Store",
    "InMemoryStore",
    "SQLiteStore",
    "Engine",
    "EventSourcedEngine",
    "Tracer",
    "NullTracer",
    "ConsoleTracer",
    "OTelTracer",
    "Memory",
    "MemoryItem",
    "NullMemory",
    "RunContext",
    "RunSpec",
    "AgentRegistry",
    "context_from_spec",
    "Budget",
    "Router",
    "SingleModel",
    "RuleRouter",
    "Guardrails",
    "Autonomy",
    "Scenario",
    "ScenarioResult",
    "Check",
    "EvalReport",
    "evaluate",
    "Judge",
    "Gate",
    "GateDecision",
    "output_contains",
    "output_satisfies",
    "forbids_tool",
    "within_steps",
    "within_tokens",
    "judged",
    "scenario_from_result",
    "tools_called",
]

__version__ = "0.0.1"
