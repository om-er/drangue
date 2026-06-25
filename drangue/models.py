"""The model seam.

The core never imports a provider directly. It talks to a `Model`: anything
with a `generate` method. The Anthropic adapter ships as the default. Swap in
your own object (or a fake, for tests) and the agent loop does not change.
"""

from __future__ import annotations

import abc
import json
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

    def tool_result_message(self, results: list[ToolResult]) -> list[dict]:
        """Build the messages carrying tool results back to the model.

        Returns a list, since some providers want one message per result.
        This default uses the content-block format Anthropic expects.
        Override if your provider needs a different shape.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.id, "content": r.content}
                for r in results
            ],
        }]


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


class OpenAIModel(Model):
    """Adapter for any OpenAI-compatible Chat Completions endpoint.

    One adapter, many cheap backends. Point `base_url` at whichever you like:

        OpenAIModel("gpt-4o-mini")                                  # OpenAI
        OpenAIModel("deepseek-chat", base_url="https://api.deepseek.com")
        OpenAIModel("llama3.1", base_url="http://localhost:11434/v1",
                    api_key="ollama")                               # Ollama, free
        OpenAIModel("llama-3.1-8b-instant",
                    base_url="https://api.groq.com/openai/v1")       # Groq

    `api_key` and `base_url` fall back to the OpenAI SDK's own defaults
    (the OPENAI_API_KEY / OPENAI_BASE_URL environment variables) when omitted.
    """

    def __init__(self, model: str, *, client: t.Any = None, api_key: str = None,
                 base_url: str = None, max_tokens: int = 4096, **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "The OpenAI SDK is required for OpenAI-compatible models. "
                    "Install it with: pip install \"drangue[openai]\""
                ) from exc
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.client = client

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        # Convert drangue's tool schema into the OpenAI function shape.
        return [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        } for tool in tools]

    def generate(self, *, system, messages, tools):
        wire_messages = list(messages)
        if system:
            wire_messages = [{"role": "system", "content": system}] + wire_messages

        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire_messages,
        }
        if tools:
            params["tools"] = self._to_openai_tools(tools)
        params.update(self.kwargs)

        resp = self.client.chat.completions.create(**params)
        message = resp.choices[0].message
        text = message.content or ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict] = []
        for tc in (message.tool_calls or []):
            args = json.loads(tc.function.arguments or "{}")
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            raw_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            })

        assistant_message: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            assistant_message["tool_calls"] = raw_tool_calls

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            stop_reason=resp.choices[0].finish_reason,
        )

    def tool_result_message(self, results):
        # OpenAI wants one message per tool result, keyed by tool_call_id.
        return [
            {"role": "tool", "tool_call_id": r.id, "content": r.content}
            for r in results
        ]
