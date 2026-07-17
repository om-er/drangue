"""Phase 4 tests: budgets, routing, model recording, and prompt caching.

All offline with scripted models and a fake Anthropic client.
"""

from types import SimpleNamespace

from drangue import (
    Agent,
    AnthropicModel,
    Budget,
    ModelResponse,
    RuleRouter,
    ToolCall,
    tool,
)
from drangue.models import Model


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class LoopingModel(Model):
    """Always asks for a tool, reporting fixed usage; loops until stopped."""

    def __init__(self, name="m", per_call_tokens=100):
        self.model = name
        self.per_call_tokens = per_call_tokens
        self.calls = 0

    async def generate(self, *, system, messages, tools, idempotency_key=None):
        self.calls += 1
        return ModelResponse(
            tool_calls=[ToolCall(f"c{self.calls}", "add", {"a": 1, "b": 1})],
            usage={"input_tokens": self.per_call_tokens, "output_tokens": 0},
        )


async def test_token_budget_stops_before_an_unaffordable_step():
    model = LoopingModel(per_call_tokens=100)
    agent = Agent(model=model, tools=[add], budget=Budget(max_tokens=150))
    result = await agent.run("go")

    # After call 1 spend is 100 (< 150, allowed). After call 2 spend is 200,
    # so the third model call is refused before it starts.
    assert model.calls == 2
    assert "budget" in result.output


async def test_usd_budget_uses_prices_and_recorded_model_name():
    model = LoopingModel(name="pricey", per_call_tokens=1_000_000)
    prices = {"pricey": {"input": 10.0, "output": 30.0}}   # $10 per 1M input tokens
    agent = Agent(model=model, tools=[add],
                  budget=Budget(max_usd=15.0, prices=prices))
    result = await agent.run("go")

    # Each call costs $10 of input. After 2 calls ($20) the next is refused.
    assert model.calls == 2
    assert "budget" in result.output


def test_usd_budget_without_prices_is_refused_at_construction():
    # A dollar limit with no price table computes spend as 0.0, so it would
    # never fire. Refuse it up front rather than fail open on the money path.
    try:
        Budget(max_usd=10.0)
        assert False, "expected ValueError for max_usd without prices"
    except ValueError as exc:
        assert "price table" in str(exc)


def test_token_budget_without_prices_is_still_fine():
    # Only the dollar limit needs prices; token budgets stand alone.
    assert Budget(max_tokens=100).prices is None


async def test_usd_budget_with_an_unpriced_model_stops_the_run_gracefully():
    # Routing can send a step to a model the table does not cover. Counting it
    # as free would under-report spend and stop the limit from firing — but
    # crashing mid-loop would leave a run that can never finish. Fail closed
    # as a recorded, graceful stop before the next model call.
    model = LoopingModel(name="unpriced", per_call_tokens=1_000_000)
    budget = Budget(max_usd=15.0, prices={"other": {"input": 10.0, "output": 30.0}})
    agent = Agent(model=model, tools=[add], budget=budget)

    result = await agent.run("go")
    assert result.output.startswith("(stopped: budget unenforceable")
    assert "no price for model 'unpriced'" in result.output
    assert any(e.type == "run_finished" for e in result.events)


async def test_result_reports_cost_given_a_price_table():
    # Phase 1's "Result reports total tokens and cost". Cost needs prices, so
    # it is a method; it shares Budget's math and so cannot disagree with it.
    model = LoopingModel(name="pricey", per_call_tokens=1_000_000)
    prices = {"pricey": {"input": 10.0, "output": 30.0}}
    agent = Agent(model=model, tools=[add], max_steps=2)
    result = await agent.run("go")

    # 2 model calls, 1M input tokens each, at $10 per 1M input = $20.
    assert result.usage["input_tokens"] == 2_000_000
    assert result.cost(prices) == 20.0

    # And it agrees with what a Budget would have charged for the same log.
    assert result.cost(prices) == Budget(max_usd=999.0, prices=prices).usd(result.events)


async def test_result_cost_refuses_an_unpriced_model():
    model = LoopingModel(name="unpriced", per_call_tokens=1_000)
    agent = Agent(model=model, tools=[add], max_steps=1)
    result = await agent.run("go")

    try:
        result.cost({"other": {"input": 1.0, "output": 1.0}})
        assert False, "expected ValueError for a model missing from the price table"
    except ValueError as exc:
        assert "no price for model 'unpriced'" in str(exc)


def test_cache_tokens_are_priced_with_anthropic_default_ratios():
    from drangue.budget import cost_from_events

    events = [SimpleNamespace(type="model_decision", payload={
        "model": "m",
        "usage": {"input_tokens": 1_000_000, "output_tokens": 0,
                  "cache_creation_input_tokens": 1_000_000,
                  "cache_read_input_tokens": 1_000_000},
    })]
    prices = {"m": {"input": 10.0, "output": 30.0}}
    # 10 (input) + 12.5 (write at 1.25x) + 1 (read at 0.1x)
    assert cost_from_events(events, prices) == 23.5


def test_cache_prices_can_be_given_explicitly():
    from drangue.budget import cost_from_events

    events = [SimpleNamespace(type="model_decision", payload={
        "model": "m",
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 1_000_000},
    })]
    # e.g. OpenAI-style cached reads at half the input price
    prices = {"m": {"input": 10.0, "output": 30.0, "cache_read": 5.0}}
    assert cost_from_events(events, prices) == 5.0


def test_token_budget_counts_cache_tokens():
    events = [SimpleNamespace(type="model_decision", payload={
        "usage": {"input_tokens": 50, "output_tokens": 10,
                  "cache_creation_input_tokens": 2_000,
                  "cache_read_input_tokens": 18_000},
    })]
    assert Budget(max_tokens=100_000).tokens(events) == 20_060


def test_cost_can_under_count_explicitly_when_asked():
    # strict=False is the documented escape hatch: approximate, never silent.
    from drangue.budget import cost_from_events

    events = [
        SimpleNamespace(type="model_decision",
                        payload={"model": "priced",
                                 "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}),
        SimpleNamespace(type="model_decision",
                        payload={"model": "unpriced",
                                 "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}),
    ]
    prices = {"priced": {"input": 10.0, "output": 30.0}}
    assert cost_from_events(events, prices, strict=False) == 10.0


async def test_model_name_is_recorded_in_the_log():
    model = LoopingModel(name="claude-test")
    agent = Agent(model=model, tools=[add], max_steps=1)
    result = await agent.run("go")
    decisions = [e for e in result.events if e.type == "model_decision"]
    assert decisions[0].payload["model"] == "claude-test"


async def test_rule_router_picks_model_per_step():
    class Named(Model):
        def __init__(self, name, steps):
            self.model = name
            self.steps = steps
            self.i = 0
            self.calls = 0

        async def generate(self, *, system, messages, tools, idempotency_key=None):
            self.calls += 1
            resp = self.steps[self.i]
            self.i += 1
            return resp

    smart = Named("smart", [ModelResponse(tool_calls=[ToolCall("c1", "add", {"a": 1, "b": 1})])])
    cheap = Named("cheap", [ModelResponse(text="done")])

    # First model call (index 0) -> smart; everything after -> cheap.
    router = RuleRouter(default=cheap, rules=[(lambda msgs, i: i == 0, smart)])
    result = await Agent(model=router, tools=[add]).run("go")

    assert result.output == "done"
    assert smart.calls == 1
    assert cheap.calls == 1
    models_used = [e.payload["model"] for e in result.events if e.type == "model_decision"]
    assert models_used == ["smart", "cheap"]


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")


async def test_cache_marks_the_stable_prefix():
    fake = FakeAnthropic()
    model = AnthropicModel("claude-test", client=fake, cache=True)
    await model.generate(system="rules", messages=[{"role": "user", "content": "hi"}],
                         tools=[add])

    sent = fake.calls[0]
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"]["type"] == "ephemeral"
    assert sent["tools"][-1]["cache_control"]["type"] == "ephemeral"


async def test_no_cache_by_default():
    fake = FakeAnthropic()
    model = AnthropicModel("claude-test", client=fake)   # cache defaults to off
    await model.generate(system="rules", messages=[{"role": "user", "content": "hi"}],
                         tools=[add])

    sent = fake.calls[0]
    assert sent["system"] == "rules"
    assert "cache_control" not in sent["tools"][-1]


def test_agent_passes_cache_through_to_a_named_model():
    # The gap this closes: AnthropicModel has always honored cache=, but the
    # facade dropped the flag, so the documented headline path could never
    # enable caching. Swap the class the facade builds to observe the handoff.
    import drangue.agent as agent_mod

    captured = {}

    class RecordingModel:
        def __init__(self, model, *, max_tokens=4096, cache=False, **kwargs):
            captured.update(model=model, max_tokens=max_tokens, cache=cache)

    original = agent_mod.AnthropicModel
    agent_mod.AnthropicModel = RecordingModel
    try:
        Agent("claude-test", tools=[add], cache=True)
    finally:
        agent_mod.AnthropicModel = original

    assert captured == {"model": "claude-test", "max_tokens": 4096, "cache": True}


def test_cache_is_refused_where_it_would_be_silently_dropped():
    # A prebuilt model or a router owns its own cache setting, so accepting the
    # flag here would leave caching looking enabled while it is not.
    for label, kwargs in [
        ("model instance", {"model": LoopingModel()}),
        ("router", {"model": RuleRouter(default=LoopingModel(), rules=[])}),
        ("explicit router=", {"model": None, "router": RuleRouter(default=LoopingModel(), rules=[])}),
    ]:
        try:
            Agent(tools=[add], cache=True, **kwargs)
            assert False, f"expected ValueError for cache=True with a {label}"
        except ValueError as exc:
            assert "cache=True only applies" in str(exc)


async def test_anthropic_usage_captures_cache_tokens():
    class FakeWithUsage:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        async def _create(self, **kwargs):
            usage = SimpleNamespace(
                input_tokens=50, output_tokens=10,
                cache_creation_input_tokens=2000,
                cache_read_input_tokens=18000,
            )
            return SimpleNamespace(content=[], usage=usage, stop_reason="end_turn")

    model = AnthropicModel("claude-test", client=FakeWithUsage(), cache=True)
    resp = await model.generate(system="rules",
                                messages=[{"role": "user", "content": "hi"}],
                                tools=[])
    assert resp.usage == {
        "input_tokens": 50,
        "output_tokens": 10,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 18000,
    }


async def test_thinking_blocks_survive_the_round_trip_to_the_next_turn():
    # With extended thinking + tool use, Anthropic requires the SIGNED
    # thinking blocks re-sent at the start of the prior assistant turn;
    # dropping them 400s the second step of every tool-using run.
    class ThinkingFake:
        def __init__(self):
            self.calls = []
            self.messages = SimpleNamespace(create=self._create)

        async def _create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                content = [
                    SimpleNamespace(type="thinking", thinking="let me add",
                                    signature="sig-abc"),
                    SimpleNamespace(type="tool_use", id="c1", name="add",
                                    input={"a": 2, "b": 3}),
                ]
            else:
                content = [SimpleNamespace(type="text", text="it is 5")]
            return SimpleNamespace(content=content, usage=None, stop_reason="tool_use")

    fake = ThinkingFake()
    model = AnthropicModel("claude-test", client=fake)
    result = await Agent(model=model, tools=[add]).run("2+3?")

    assert result.output == "it is 5"
    # The second request's assistant turn starts with the signed thinking block.
    second = fake.calls[1]["messages"]
    assistant = [m for m in second if m["role"] == "assistant"][0]
    assert assistant["content"][0] == {
        "type": "thinking", "thinking": "let me add", "signature": "sig-abc",
    }
    assert assistant["content"][1]["type"] == "tool_use"
    # And the blocks are recorded in the log, so a RESUMED run re-renders the
    # same conversation.
    decision = [e for e in result.events if e.type == "model_decision"][0]
    assert decision.payload["thinking_blocks"][0]["signature"] == "sig-abc"


async def test_anthropic_does_not_send_an_idempotency_header():
    # Anthropic has no request idempotency key; the param is accepted for
    # interface parity but must not be sent (it would be a silent no-op).
    fake = FakeAnthropic()
    model = AnthropicModel("claude-test", client=fake)
    await model.generate(system="rules", messages=[{"role": "user", "content": "hi"}],
                         tools=[add], idempotency_key="run-1:1")

    headers = fake.calls[0].get("extra_headers") or {}
    assert "Idempotency-Key" not in headers
