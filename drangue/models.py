"""The model seam.

The core never imports a provider directly. It talks to a `Model`: anything
with an async `generate` method. Adapters take a provider-neutral conversation
history and render it into their own wire format internally, so neither the
orchestrator nor the executor knows anything provider-specific.

Neutral message shapes (what the orchestrator produces):
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{id, name, arguments}]}
    {"role": "tool", "call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

import abc
import json
import typing as t
from dataclasses import dataclass, field

from .hardening import MALFORMED_ARGS_KEY


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """One turn of model output, normalized across providers."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    reasoning: str | None = None    # the model's stated intent, when available
    stop_reason: str | None = None
    # Provider-opaque reasoning blocks that must be re-sent verbatim on the
    # next turn (Anthropic's signed thinking blocks). JSON-shaped so they
    # survive the event log; adapters that have none leave this empty.
    thinking_blocks: list = field(default_factory=list)


class Model(abc.ABC):
    """Implement async `generate` and you are a drangue model."""

    @abc.abstractmethod
    async def generate(self, *, system: str, messages: list[dict], tools: list,
                       idempotency_key: str | None = None) -> ModelResponse:
        ...


class AnthropicModel(Model):
    """Default adapter for Claude models via the async Anthropic SDK."""

    def __init__(self, model: str, *, client: t.Any = None,
                 max_tokens: int = 4096, cache: bool = False, **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.cache = cache
        self.kwargs = kwargs
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "The Anthropic SDK is required for the default model. "
                    "Install it with: pip install \"drangue[anthropic]\""
                ) from exc
            client = anthropic.AsyncAnthropic()
        self.client = client

    @staticmethod
    def _render_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        pending_tool_blocks: list[dict] = []

        def flush():
            nonlocal pending_tool_blocks
            if pending_tool_blocks:
                out.append({"role": "user", "content": pending_tool_blocks})
                pending_tool_blocks = []

        for m in messages:
            if m["role"] == "tool":
                pending_tool_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": m["call_id"],
                    "content": m["content"],
                })
                continue
            flush()
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                # With extended thinking enabled, the API requires the signed
                # thinking blocks to be re-sent at the START of the prior
                # assistant turn when tool results follow; dropping them is a
                # guaranteed 400 on the next step.
                blocks.extend(m.get("thinking_blocks") or [])
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["name"],
                        "input": c["arguments"],
                    })
                # Anthropic rejects an empty content array. An assistant turn
                # with neither text nor tool calls is terminal today (it never
                # gets re-sent), but guard against it rather than leave the trap.
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        flush()
        return out

    async def generate(self, *, system, messages, tools, idempotency_key=None,
                       on_delta=None):
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._render_messages(messages),
        }
        if system:
            # Context is ordered stable-to-volatile: system and tools first (they
            # never change within a run), conversation after. With cache on, mark
            # that stable prefix so the provider can reuse it across steps.
            if self.cache:
                params["system"] = [{
                    "type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                params["system"] = system
        if tools:
            schemas = [tool.to_schema() for tool in tools]
            if self.cache and schemas:
                schemas[-1] = {**schemas[-1], "cache_control": {"type": "ephemeral"}}
            params["tools"] = schemas
        params.update(self.kwargs)

        # NOTE: Anthropic's Messages API does not support a request-level
        # idempotency key (unlike OpenAI). `idempotency_key` is accepted for
        # interface parity but deliberately NOT sent: a custom header would be a
        # silent no-op. Consequence: the model call is AT-LEAST-ONCE on resume.
        # If a crash lands between this call returning and the decision being
        # appended, a resumed run re-calls the model (possible double charge and
        # divergence). Make downstream effects idempotent instead (tools do this
        # via the idempotency_key parameter, see hardening.py).

        if on_delta is not None:
            # Stream text deltas to the caller as they arrive; the final
            # message (the recorded fact) is identical to the blocking path.
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    await on_delta(text)
                resp = await stream.get_final_message()
        else:
            resp = await self.client.messages.create(**params)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_blocks: list[dict] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(getattr(block, "thinking", ""))
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", None),
                })
            elif block.type == "redacted_thinking":
                thinking_blocks.append({
                    "type": "redacted_thinking",
                    "data": getattr(block, "data", ""),
                })
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        # Anthropic reports cached tokens SEPARATELY from input_tokens, billed
        # at different rates (writes 1.25x, reads 0.1x). Dropping them would
        # hide the bulk of prompt spend from budgets on exactly the runs that
        # enable cache=True. Usage keys are disjoint counts: input_tokens is
        # uncached input only.
        u = getattr(resp, "usage", None)
        usage = None
        if u:
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
            for extra in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                v = getattr(u, extra, None)
                if v:
                    usage[extra] = v

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            reasoning="".join(reasoning_parts) or None,
            stop_reason=resp.stop_reason,
            thinking_blocks=thinking_blocks,
        )


class OpenAIModel(Model):
    """Adapter for any OpenAI-compatible Chat Completions endpoint.

    One adapter, many cheap backends. Point `base_url` at whichever you like:

        OpenAIModel("gpt-4o-mini")                                  # OpenAI
        OpenAIModel("deepseek-chat", base_url="https://api.deepseek.com")
        OpenAIModel("llama3.1", base_url="http://localhost:11434/v1",
                    api_key="ollama")                               # Ollama, free
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
            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.client = client

    @staticmethod
    def _to_openai_tools(schemas: list[dict]) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        } for s in schemas]

    @staticmethod
    def _render_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                    } for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m["call_id"],
                    "content": m["content"],
                })
        return out

    # Model families that reject the legacy `max_tokens` parameter and require
    # `max_completion_tokens`. Third-party OpenAI-compatible backends (Ollama,
    # DeepSeek, Groq, ...) still expect `max_tokens`, so this stays a prefix
    # check on OpenAI's own reasoning-model names rather than a blanket switch.
    _MAX_COMPLETION_TOKENS_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    @staticmethod
    def _parse_arguments(raw: str) -> dict:
        raw = raw or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            return {MALFORMED_ARGS_KEY: raw}
        if not isinstance(arguments, dict):
            return {MALFORMED_ARGS_KEY: raw}
        return arguments

    @staticmethod
    def _usage_from(u) -> dict | None:
        # OpenAI's prompt_tokens INCLUDES cached tokens; split them out so the
        # usage dict has the same invariant as the Anthropic adapter's: keys
        # are disjoint counts, input_tokens is uncached input only.
        if not u:
            return None
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) if details else 0) or 0
        usage = {
            "input_tokens": u.prompt_tokens - cached,
            "output_tokens": u.completion_tokens,
        }
        if cached:
            usage["cache_read_input_tokens"] = cached
        return usage

    async def _generate_streaming(self, params: dict, on_delta) -> ModelResponse:
        """Drive a streamed completion, forwarding text deltas as they arrive."""
        params = {**params, "stream": True,
                  # Without this the final usage never arrives on the stream.
                  "stream_options": {"include_usage": True}}
        text_parts: list[str] = []
        acc: dict[int, dict] = {}       # tool-call fragments, keyed by index
        finish_reason = None
        usage_obj = None

        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                await on_delta(content)
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = acc.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        tool_calls = [
            ToolCall(id=slot["id"], name=slot["name"],
                     arguments=self._parse_arguments(slot["arguments"]))
            for _, slot in sorted(acc.items())
        ]
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=self._usage_from(usage_obj),
            stop_reason=finish_reason,
        )

    async def generate(self, *, system, messages, tools, idempotency_key=None,
                       on_delta=None):
        token_param = (
            "max_completion_tokens"
            if self.model.lower().startswith(self._MAX_COMPLETION_TOKENS_PREFIXES)
            else "max_tokens"
        )
        params: dict = {
            "model": self.model,
            token_param: self.max_tokens,
            "messages": self._render_messages(system, messages),
        }
        if tools:
            params["tools"] = self._to_openai_tools([tool.to_schema() for tool in tools])
        params.update(self.kwargs)
        if "max_completion_tokens" in params:
            # A caller passing the newer param (or a matched prefix) must not
            # also send the legacy one; the API rejects requests with both.
            params.pop("max_tokens", None)

        # Request-level idempotency (OpenAI-compatible backends honor this header):
        # if the process dies after the call returns but before the decision is
        # appended, a resumed run sends the same key and the provider returns the
        # same response (within its idempotency window) instead of charging twice
        # or diverging. This makes the model call effectively exactly-once on the
        # OpenAI path, unlike the Anthropic path (see AnthropicModel.generate).
        if idempotency_key:
            headers = dict(params.get("extra_headers") or {})
            headers["Idempotency-Key"] = idempotency_key
            params["extra_headers"] = headers

        if on_delta is not None:
            return await self._generate_streaming(params, on_delta)

        resp = await self.client.chat.completions.create(**params)
        message = resp.choices[0].message
        text = message.content or ""

        tool_calls: list[ToolCall] = []
        for tc in (message.tool_calls or []):
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=self._parse_arguments(tc.function.arguments),
            ))

        usage = self._usage_from(getattr(resp, "usage", None))

        # NOTE: `reasoning` is deliberately left unset. The Chat Completions API
        # exposes no equivalent of Anthropic's thinking blocks, and inventing one
        # by echoing `text` would put a summary where callers expect the model's
        # actual stated intent. Consequence: on this adapter an assisted-mode
        # approval surface shows `reasoning: None` and falls back to the
        # decision's text (see rollout.last_reasoning).
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=resp.choices[0].finish_reason,
        )
