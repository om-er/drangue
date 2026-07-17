"""Token streaming: live deltas through agent.stream, log unchanged.

Both adapters stream against fake clients, fully offline. The invariant under
test: deltas are ephemeral (emitted, never stored), and the recorded
model_decision is identical to what the blocking path would have produced.
"""

from types import SimpleNamespace

from drangue import Agent, AnthropicModel, ModelResponse, OpenAIModel, tool
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class FakeAnthropicStreaming:
    """Supports both create() and the stream() context manager."""

    def __init__(self, chunks, final_text):
        self.chunks = chunks
        self.final_text = final_text
        self.stream_calls = 0
        self.create_calls = 0
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    async def _create(self, **kwargs):
        self.create_calls += 1
        return self._final()

    def _final(self):
        block = SimpleNamespace(type="text", text=self.final_text)
        return SimpleNamespace(content=[block], usage=None, stop_reason="end_turn")

    def _stream(self, **kwargs):
        self.stream_calls += 1
        fake = self

        class _Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            @property
            def text_stream(self):
                async def gen():
                    for c in fake.chunks:
                        yield c
                return gen()

            async def get_final_message(self):
                return fake._final()

        return _Stream()


async def test_anthropic_streams_deltas_and_records_the_final_decision():
    client = FakeAnthropicStreaming(chunks=["hel", "lo"], final_text="hello")
    agent = Agent(model=AnthropicModel("claude-test", client=client), tools=[])

    deltas, decisions = [], []
    async for ev in agent.stream("hi", run_id="r-stream"):
        if ev.type == "model_delta":
            deltas.append(ev.payload["text"])
        elif ev.type == "model_decision":
            decisions.append(ev)

    assert deltas == ["hel", "lo"]
    assert client.stream_calls == 1 and client.create_calls == 0
    assert decisions[0].payload["text"] == "hello"
    # Ephemeral: nothing delta-shaped lands in the store.
    stored = await agent.store.load("r-stream")
    assert not [e for e in stored if e.type == "model_delta"]


async def test_plain_run_uses_the_blocking_path():
    client = FakeAnthropicStreaming(chunks=["never"], final_text="hello")
    agent = Agent(model=AnthropicModel("claude-test", client=client), tools=[])
    result = await agent.run("hi")
    assert result.output == "hello"
    assert client.create_calls == 1 and client.stream_calls == 0


def _chunk(content=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tc_frag(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


class FakeOpenAIStreaming:
    def __init__(self, script):
        # script: list of lists of chunks, one list per create() call
        self.script = script
        self.calls = []
        self.i = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self.script[self.i]
        self.i += 1

        async def gen():
            for c in chunks:
                yield c

        return gen()


async def test_openai_streaming_assembles_tool_calls_from_fragments():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                            prompt_tokens_details=None)
    script = [
        [  # first model turn: a tool call arrives in fragments
            _chunk(tool_calls=[_tc_frag(0, call_id="c1", name="add", arguments='{"a": 2')]),
            _chunk(tool_calls=[_tc_frag(0, arguments=', "b": 3}')]),
            _chunk(finish="tool_calls"),
            SimpleNamespace(choices=[], usage=usage),
        ],
        [  # second turn: streamed text answer
            _chunk(content="the answer "),
            _chunk(content="is 5", finish="stop"),
            SimpleNamespace(choices=[], usage=usage),
        ],
    ]
    client = FakeOpenAIStreaming(script)
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])

    deltas = []
    final = None
    async for ev in agent.stream("2+3?", run_id="r-oai"):
        if ev.type == "model_delta":
            deltas.append(ev.payload["text"])
        elif ev.type == "run_finished":
            final = ev.payload["output"]

    assert final == "the answer is 5"
    assert deltas == ["the answer ", "is 5"]
    # The streamed requests asked for usage on the stream.
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    # The assembled tool call executed for real.
    stored = await agent.store.load("r-oai")
    tool_result = [e for e in stored if e.type == "tool_result"][0]
    assert tool_result.payload["content"] == "5"
    decision = [e for e in stored if e.type == "model_decision"][0]
    assert decision.payload["tool_calls"][0]["arguments"] == {"a": 2, "b": 3}
    assert decision.payload["usage"] == {"input_tokens": 10, "output_tokens": 5}


async def test_custom_model_without_on_delta_still_streams_events():
    class PlainModel(Model):
        async def generate(self, *, system, messages, tools, idempotency_key=None):
            return ModelResponse(text="done")

    agent = Agent(model=PlainModel(), tools=[])
    types = [ev.type async for ev in agent.stream("go")]
    assert "model_decision" in types and "run_finished" in types
    assert "model_delta" not in types   # no support, no deltas, no crash
