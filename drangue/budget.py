"""Per-run cost budgets (Chapter 10).

Cost-predictability is a production requirement: a run's spend must be bounded,
not discovered after the fact. A Budget is checked before each expensive step
(every model call) against what the run has already spent, read from the
recorded usage in the log. Because it reads recorded facts, the check is
deterministic and replay-safe.

Token budgets always work. A dollar budget also needs a price table and the
model name recorded per step (the executor records it), since routing can send
different steps to different models.

Caveat: this is a soft ceiling, not a hard one. `exceeded` only sees usage that
is already recorded, so the step that pushes a run over the limit still runs; the
budget stops the NEXT step. Overshoot is bounded by one model call. For a hard
ceiling you would pre-estimate the next step's cost before allowing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    max_tokens: int | None = None     # total input + output tokens for the run
    max_usd: float | None = None
    # {model_name: {"input": usd_per_million, "output": usd_per_million}}
    prices: dict | None = None

    def tokens(self, events) -> int:
        total = 0
        for e in events:
            if e.type == "model_decision":
                u = e.payload.get("usage")
                if u:
                    total += u.get("input_tokens", 0) + u.get("output_tokens", 0)
        return total

    def usd(self, events) -> float:
        if not self.prices:
            return 0.0
        total = 0.0
        for e in events:
            if e.type != "model_decision":
                continue
            u = e.payload.get("usage")
            price = self.prices.get(e.payload.get("model"))
            if u and price:
                total += u.get("input_tokens", 0) / 1e6 * price.get("input", 0.0)
                total += u.get("output_tokens", 0) / 1e6 * price.get("output", 0.0)
        return total

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
