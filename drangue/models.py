"""The model seam.

The core never imports a provider directly. It talks to a `Model`: anything
with a `generate` method. The Anthropic adapter ships as the default. Swap in
your own object (or a fake, for tests) and the agent loop does not change.
"""

from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The string result of running one tool, tied back to its call."""

    id: str
    content: str


@dataclass
class ModelResponse:
    """One turn of model output, normalized across providers.

    `assistant_message` is the provider-native message to append back to the
    running history. The agent treats it as opaque.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict | None = None
    stop_reason: str | None = None


class Model(abc.ABC):
    """Implement `generate` and you are a drangue model."""

    @abc.abstractmethod
    def generate(self, *, system: str, messages: list[dict],
                 tools: list[dict]) -> ModelResponse:
        ...

    def tool_result_message(self, results: list[ToolResult]) -> dict:
        """Build the next user message carrying tool results.

        Uses the content-block format that is drangue's lingua franca.
        Override if your provider needs a different shape.
        """
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.id, "content": r.content}
                for r in results
            ],
        }


class AnthropicModel(Model):
    """Default adapter for Claude models via the Anthropic SDK."""

    def __init__(self, model: str, *, client: t.Any = None,
                 max_tokens: int = 4096, **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "The Anthropic SDK is required for the default model. "
                    "Install it with: pip install anthropic"
                ) from exc
            client = anthropic.Anthropic()
        self.client = client

    def generate(self, *, system, messages, tools):
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        params.update(self.kwargs)

        resp = self.client.messages.create(**params)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": content_blocks},
            stop_reason=resp.stop_reason,
        )
