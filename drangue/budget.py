"""Per-run cost budgets (Chapter 10).

Cost-predictability is a production requirement: a run's spend must be bounded,
not discovered after the fact. A Budget is checked before each expensive step
(every model call) against what the run has already spent, read from the
recorded usage in the log. Because it reads recorded facts, the check is
deterministic and replay-safe.

Token budgets work whenever the model reports usage; a custom Model that
records no usage contributes zero, so a token ceiling is only as good as the
adapter's accounting (both shipped adapters report it). A dollar budget also
needs a price table and the model name recorded per step (the executor records
it), since routing can send different steps to different models. Both ways of
losing a dollar budget are refused rather than absorbed: `max_usd` without
`prices` raises at construction, and a step whose model is missing from the
table is refused when spend is computed — the engine turns that refusal into a
graceful "budget unenforceable" stop before the next model call, not a crash.
An unenforceable cost control is worse than none, because it looks armed.

Caveat: this is a soft ceiling, not a hard one. `exceeded` only sees usage that
is already recorded, so the step that pushes a run over the limit still runs; the
budget stops the NEXT step. Overshoot is bounded by one model call. For a hard
ceiling you would pre-estimate the next step's cost before allowing it.

The budget gates model steps, not tools. A pending tool call still runs when the
run is already over budget (finish in-flight work; tools are cheap and do not
consume model tokens). It is the next model step that is refused, so "budget
exhausted" does not interrupt tool side effects mid-flight.
"""

from __future__ import annotations

from dataclasses import dataclass


def cost_from_events(events, prices: dict | None, *, strict: bool = True) -> float:
    """Dollar spend implied by the usage recorded in the log.

    Shared by `Budget.usd` (enforcement) and `Result.cost` (reporting) so the
    two can never disagree about what a run cost.

    `prices` maps model name -> {"input": usd_per_million, "output": ...}. The
    executor records the model per step, so a routed run prices each step
    against the model that actually ran.

    Cached prompt tokens (recorded by the adapters as
    `cache_creation_input_tokens` / `cache_read_input_tokens`, disjoint from
    `input_tokens`) are priced with the optional "cache_write" / "cache_read"
    keys. When absent they default to Anthropic's ratios (1.25x and 0.1x the
    input price). Other providers bill differently — OpenAI cached reads are
    typically 0.5x — so supply the explicit keys when it matters.

    strict=True (the default) raises when a step used a model the price table
    does not cover. Skipping it would count that model as free and silently
    under-report spend, which is how a dollar limit fails to fire. Pass
    strict=False only when an approximate figure is genuinely acceptable.
    """
    prices = prices or {}
    total = 0.0
    for e in events:
        if e.type != "model_decision":
            continue
        u = e.payload.get("usage")
        if not u:
            continue
        model = e.payload.get("model")
        price = prices.get(model)
        if price is None:
            if strict:
                hint = (
                    " (model is None: give your custom Model a `.model` name "
                    "attribute so its steps can be priced)" if model is None else ""
                )
                raise ValueError(
                    f"no price for model {model!r}, so its spend would count as "
                    f"$0. Add it to the price table (known: {sorted(prices)}), "
                    f"or pass strict=False to accept an under-count.{hint}"
                )
            continue
        p_in = price.get("input", 0.0)
        total += u.get("input_tokens", 0) / 1e6 * p_in
        total += u.get("output_tokens", 0) / 1e6 * price.get("output", 0.0)
        total += (u.get("cache_creation_input_tokens", 0) / 1e6
                  * price.get("cache_write", p_in * 1.25))
        total += (u.get("cache_read_input_tokens", 0) / 1e6
                  * price.get("cache_read", p_in * 0.1))
    return total


@dataclass
class Budget:
    max_tokens: int | None = None     # total input + output tokens for the run
    max_usd: float | None = None
    # {model_name: {"input": usd_per_million, "output": usd_per_million}}
    prices: dict | None = None

    def __post_init__(self):
        # A dollar limit without a price table is unenforceable: spend would
        # compute as 0.0 and the limit would never fire. Refuse at construction
        # rather than fail open at runtime on the money path.
        if self.max_usd is not None and not self.prices:
            raise ValueError(
                "Budget(max_usd=...) requires a price table, e.g. "
                'prices={"claude-opus-4-8": {"input": 15.0, "output": 75.0}} '
                "(USD per million tokens). Without it, spend cannot be computed "
                "and the limit would silently never fire."
            )

    def tokens(self, events) -> int:
        total = 0
        for e in events:
            if e.type == "model_decision":
                u = e.payload.get("usage")
                if u:
                    # Cached tokens are still tokens the model processed; a
                    # token ceiling that ignored them would under-fire on
                    # exactly the runs that enable caching.
                    total += (u.get("input_tokens", 0)
                              + u.get("output_tokens", 0)
                              + u.get("cache_creation_input_tokens", 0)
                              + u.get("cache_read_input_tokens", 0))
        return total

    def usd(self, events) -> float:
        # Enforcement is strict: an unpriced model must not be counted as free.
        return cost_from_events(events, self.prices)

    def exceeded(self, events) -> bool:
        if self.max_tokens is not None and self.tokens(events) >= self.max_tokens:
            return True
        if self.max_usd is not None and self.usd(events) >= self.max_usd:
            return True
        return False

    def remaining_tokens(self, events) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens(events))
