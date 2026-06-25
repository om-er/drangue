"""drangue: a tiny, obvious agent runtime for Python.

The simple path is one object and one call. Underneath, runs are driven as an
event-sourced loop (orchestrator decides, executor acts, store records) so the
same facade grows into durability, observability, and recovery without changing.
"""

from .agent import Agent
from .engine import Engine, EventSourcedEngine
from .events import Event, Result
from .models import AnthropicModel, Model, ModelResponse, OpenAIModel, ToolCall
from .store import InMemoryStore, Store
from .tool import Tool, tool

__all__ = [
    "Agent",
    "tool",
    "Tool",
    "Event",
    "Result",
    "Model",
    "AnthropicModel",
    "OpenAIModel",
    "ModelResponse",
    "ToolCall",
    "Store",
    "InMemoryStore",
    "Engine",
    "EventSourcedEngine",
]

__version__ = "0.0.1"
