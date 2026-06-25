"""drangue: a tiny, obvious agent runtime for Python.

An agent is just a model plus tools. Running it is one call.
No graphs, no chains, no base classes to inherit.
"""

from .agent import Agent, Event, Result
from .models import AnthropicModel, Model, ModelResponse, ToolCall, ToolResult
from .tool import Tool, tool

__all__ = [
    "Agent",
    "Event",
    "Result",
    "tool",
    "Tool",
    "Model",
    "AnthropicModel",
    "ModelResponse",
    "ToolCall",
    "ToolResult",
]

__version__ = "0.0.1"
