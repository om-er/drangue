"""Offline tests for the OpenAI-compatible adapter.

A fake async client stands in for the OpenAI SDK, so these exercise the real
render and parse code without a network call or API key.
"""

from types import SimpleNamespace

from drangue import Agent, OpenAIModel, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _resp(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _tc(call_id, name, arguments_json):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


class FakeOpenAIClient:
    """Mimics the slice of the async OpenAI SDK that OpenAIModel touches."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.i = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        resp = self.responses[self.i]
        self.i += 1
        return resp


async def test_openai_tool_loop():
    client = FakeOpenAIClient([
        _resp(tool_calls=[_tc("c1", "add", '{"a": 2, "b": 3}')]),
        _resp(content="The answer is 5."),
    ])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])
    result = await agent.run("what is 2 + 3?")

    assert result.output == "The answer is 5."
    assert result.steps == 2


async def test_tools_are_converted_to_openai_function_shape():
    client = FakeOpenAIClient([_resp(content="hi")])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])
    await agent.run("hello")

    sent = client.calls[0]["tools"][0]
    assert sent["type"] == "function"
    assert sent["function"]["name"] == "add"
    assert sent["function"]["parameters"]["properties"]["a"]["type"] == "integer"


async def test_system_prompt_becomes_a_system_message():
    client = FakeOpenAIClient([_resp(content="ok")])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add],
                  instructions="Be brief.")
    await agent.run("hello")

    first = client.calls[0]["messages"][0]
    assert first == {"role": "system", "content": "Be brief."}


async def test_idempotency_key_is_sent_as_a_header():
    client = FakeOpenAIClient([_resp(content="ok")])
    model = OpenAIModel("test", client=client)
    await model.generate(system="", messages=[{"role": "user", "content": "hi"}],
                         tools=[], idempotency_key="run-1:1")

    assert client.calls[0]["extra_headers"]["Idempotency-Key"] == "run-1:1"


async def test_reasoning_models_get_max_completion_tokens():
    client = FakeOpenAIClient([_resp(content="ok")])
    model = OpenAIModel("o3-mini", client=client)
    await model.generate(system="", messages=[{"role": "user", "content": "hi"}],
                         tools=[])

    sent = client.calls[0]
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 4096


async def test_explicit_max_completion_tokens_drops_the_legacy_param():
    client = FakeOpenAIClient([_resp(content="ok")])
    model = OpenAIModel("some-proxy-model", client=client,
                        max_completion_tokens=512)
    await model.generate(system="", messages=[{"role": "user", "content": "hi"}],
                         tools=[])

    sent = client.calls[0]
    assert "max_tokens" not in sent          # never send both; the API 400s
    assert sent["max_completion_tokens"] == 512


async def test_non_reasoning_models_keep_max_tokens():
    client = FakeOpenAIClient([_resp(content="ok")])
    model = OpenAIModel("llama3.1", client=client)
    await model.generate(system="", messages=[{"role": "user", "content": "hi"}],
                         tools=[])

    assert client.calls[0]["max_tokens"] == 4096


async def test_malformed_tool_arguments_become_a_clean_failure():
    import json

    client = FakeOpenAIClient([
        _resp(tool_calls=[_tc("c1", "add", '{"a": 2, "b": ')]),   # truncated JSON
        _resp(content="recovered"),
    ])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])
    result = await agent.run("go")

    assert result.output == "recovered"      # the run survives
    tool_result = [e for e in result.events if e.type == "tool_result"][0]
    parsed = json.loads(tool_result.payload["content"])
    assert parsed["ok"] is False
    assert parsed["error"]["category"] == "invalid_arguments"
    assert '{"a": 2, "b": ' in parsed["error"]["message"]


async def test_non_object_tool_arguments_become_a_clean_failure():
    import json

    client = FakeOpenAIClient([
        _resp(tool_calls=[_tc("c1", "add", '[1, 2]')]),   # valid JSON, wrong shape
        _resp(content="recovered"),
    ])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])
    result = await agent.run("go")

    assert result.output == "recovered"
    tool_result = [e for e in result.events if e.type == "tool_result"][0]
    assert json.loads(tool_result.payload["content"])["error"]["category"] == "invalid_arguments"


async def test_openai_usage_splits_cached_tokens_out_of_input():
    message = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=20_050, completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=20_000),
    )
    client = FakeOpenAIClient([SimpleNamespace(choices=[choice], usage=usage)])
    model = OpenAIModel("test", client=client)
    resp = await model.generate(system="", messages=[{"role": "user", "content": "hi"}],
                                tools=[])
    assert resp.usage == {
        "input_tokens": 50,                       # uncached only
        "output_tokens": 10,
        "cache_read_input_tokens": 20_000,
    }


async def test_tool_results_use_the_tool_role_keyed_by_call_id():
    client = FakeOpenAIClient([
        _resp(tool_calls=[_tc("c1", "add", '{"a": 1, "b": 1}')]),
        _resp(content="two"),
    ])
    agent = Agent(model=OpenAIModel("test", client=client), tools=[add])
    await agent.run("go")

    second_messages = client.calls[1]["messages"]
    tool_msg = [m for m in second_messages if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "2"
