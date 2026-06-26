"""Model routing (Chapter 10).

Send each step to the cheapest model that can handle it. A judgment step goes to
a capable, expensive model; a mechanical step can go to a cheap one, with no
quality loss on the trivial decision. Structural routing (decide by the step,
in advance) is more reliable than dynamic escalation.

A Router just answers: which model for this model step? The default wraps a
single model. RuleRouter picks by a predicate over the conversation and the
step index. The executor records which model actually ran, so routing is visible
in the trace and accountable in the budget.
"""

from __future__ import annotations

import typing as t


class Router(t.Protocol):
    def choose(self, *, messages: list, step_index: int): ...


class SingleModel:
    """Always the same model. The default."""

    def __init__(self, model):
        self.model = model

    def choose(self, *, messages: list, step_index: int):
        return self.model


class RuleRouter:
    """Pick a model by rules over (messages, step_index).

    `rules` is a sequence of (predicate, model). The first predicate that returns
    true wins; otherwise `default` is used.

        router = RuleRouter(
            default=smart,
            rules=[(lambda msgs, i: i > 0, cheap)],   # only the first step is "judgment"
        )
    """

    def __init__(self, default, rules=()):
        self.default = default
        self.rules = list(rules)

    def choose(self, *, messages: list, step_index: int):
        for predicate, model in self.rules:
            if predicate(messages, step_index):
                return model
        return self.default
