"""drangue: a tiny, obvious agent runtime for Python.

The simple path is one object and one call. Underneath, runs are driven as an
event-sourced loop (orchestrator decides, executor acts, store records) so the
same facade grows into durability, observability, and recovery without changing.
"""

from .agent import Agent
from .engine import Engine, EventSourcedEngine
from .events import Event, Result, Span
from .memory import Memory, MemoryItem, NullMemory
from .models import AnthropicModel, Model, ModelResponse, OpenAIModel, ToolCall
from .observability import ConsoleTracer, NullTracer, OTelTracer, Tracer
from .store import InMemoryStore, SQLiteStore, Store
from .tool import Tool, tool

__all__ = [
    "Agent",
    "tool",
    "Tool",
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
]

__version__ = "0.0.1"
