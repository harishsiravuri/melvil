"""Confusion-driven reflection (feature 1 of the classification layer).

As the optimizer runs, every full dev evaluation updates a `ConfusionState`:
the dev confusion matrix, the top confused label pairs with concrete
misclassified examples, and per-example miss counts (consumed by hard-example
mining). Two consumers:

- `ConfusionComponentSelector` — a gepa `module_selector` that points each
  reflection round at the codebook components implicated in the currently
  worst confusion (both sides of the boundary), instead of round-robin.
- the proposer's confusion block — `render_confusion_block()` puts the top
  confused pairs and examples in front of the reflection LM with an explicit
  instruction to sharpen those boundaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from melvil.artifact import BOUNDARY_COMPONENT, TASK_COMPONENT, label_component
from melvil.program import Prediction


def confusion_matrix(preds: list[Prediction], label_names: list[str]) -> dict[str, Any]:
    idx = {name: i for i, name in enumerate(label_names)}
    n = len(label_names)
    matrix = [[0] * (n + 1) for _ in range(n)]  # extra column: unparseable
    for p in preds:
        gi = idx.get(p.gold)
        if gi is None:
            continue
        pi = idx.get(p.predicted, n)
        matrix[gi][pi] += 1
    return {"labels": list(label_names), "matrix": matrix, "unparseable_column": True}


def top_confused_pairs(preds: list[Prediction], k: int = 3) -> list[tuple[str, str, int]]:
    """Most frequent (gold, predicted) error pairs. Unparseable outputs count
    under predicted='' and are reported too (they indicate format problems)."""
    counts: Counter[tuple[str, str]] = Counter()
    for p in preds:
        if not p.correct:
            counts[(p.gold, p.predicted)] += 1
    return [(g, pr, c) for (g, pr), c in counts.most_common(k)]


@dataclass
class ConfusionState:
    """Rolling view of dev-set behavior across the run."""

    rounds: int = 0
    last_preds: list[Prediction] = field(default_factory=list)
    last_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    miss_counts: dict[str, int] = field(default_factory=dict)  # example text -> misses
    miss_pairs: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def update(self, preds: list[Prediction], k: int = 5) -> None:
        self.rounds += 1
        self.last_preds = preds
        self.last_pairs = top_confused_pairs(preds, k)
        for p in preds:
            if not p.correct:
                self.miss_counts[p.text] = self.miss_counts.get(p.text, 0) + 1
                self.miss_pairs[p.text][(p.gold, p.predicted)] += 1

    def examples_for_pair(self, gold: str, predicted: str, limit: int = 3) -> list[Prediction]:
        return [
            p for p in self.last_preds
            if not p.correct and p.gold == gold and p.predicted == predicted
        ][:limit]


class ConfusionComponentSelector:
    """gepa `module_selector`: schedule reflection rounds across the confused
    label boundaries with deficit-weighted round-robin (stride scheduling), so
    heavily-confused pairs get proportionally more updates but no confused pair
    starves. (v0.1 greedily picked the single worst pair every round, which in
    practice sent nearly all updates to two labels — trace-verified in the v0.1
    benchmarks.) Every SHARED_EVERY-th call goes to the shared components
    (boundary rules / task instruction); falls back to round-robin over all
    components until confusion data exists."""

    #: every Nth reflection round goes to the shared components
    SHARED_EVERY = 4
    #: how many of the most confused pairs participate in the rotation
    TOP_K = 5

    def __init__(self, state: ConfusionState, all_components: list[str]):
        self.state = state
        self.all_components = all_components
        self.calls = 0
        self._credits: dict[tuple[str, str], float] = {}

    def _current_pairs(self, trajectories) -> list[tuple[str, str, int]]:
        combined: Counter[tuple[str, str]] = Counter(
            {(g, pr): c for g, pr, c in self.state.last_pairs}
        )
        if trajectories:  # fresh minibatch signal
            for t in trajectories:
                p = t.get("prediction") if isinstance(t, dict) else None
                if p is not None and not p.correct:
                    combined[(p.gold, p.predicted)] += 1
        return [(g, pr, c) for (g, pr), c in combined.most_common(self.TOP_K)]

    def __call__(
        self, gepa_state, trajectories, subsample_scores, candidate_idx, candidate
    ) -> list[str]:
        self.calls += 1
        if self.calls % self.SHARED_EVERY == 0:
            shared = [c for c in (BOUNDARY_COMPONENT, TASK_COMPONENT) if c in candidate]
            if shared:
                return [shared[(self.calls // self.SHARED_EVERY) % len(shared)]]

        eligible = [
            (g, pr, c) for g, pr, c in self._current_pairs(trajectories)
            if label_component(g) in candidate or label_component(pr) in candidate
        ]
        if not eligible:  # no confusion data yet -> round-robin over everything
            return [self.all_components[self.calls % len(self.all_components)]]

        # stride scheduling: each pair accrues credit proportional to its error
        # count; the pair with the most accumulated credit is served and pays
        # the full round's weight. Guarantees proportional, starvation-free
        # rotation among the TOP_K pairs.
        keys = {(g, pr) for g, pr, _ in eligible}
        self._credits = {k: v for k, v in self._credits.items() if k in keys}
        total = sum(c for *_, c in eligible)
        for g, pr, c in eligible:
            self._credits[(g, pr)] = self._credits.get((g, pr), 0.0) + c
        best = max(self._credits, key=lambda k: self._credits[k])
        self._credits[best] -= total
        comps = [c for c in (label_component(best[0]), label_component(best[1])) if c in candidate]
        return comps or [self.all_components[self.calls % len(self.all_components)]]


def render_confusion_block(state: ConfusionState, max_pairs: int = 3) -> str:
    if not state.last_pairs:
        return ""
    lines = [
        "Dev-set confusion analysis for the CURRENT prompt (most frequent error "
        "patterns; an empty predicted label means the response failed to parse):"
    ]
    for gold, predicted, count in state.last_pairs[:max_pairs]:
        pred_desc = predicted if predicted else "(unparseable output)"
        lines.append(f"\n- true '{gold}' misread as '{pred_desc}' ({count}x). Examples:")
        for p in state.examples_for_pair(gold, predicted):
            lines.append(f"    Text: {p.text[:220]}")
            lines.append(f"    Model output: {p.raw[:80]}")
    lines.append(
        "\nWhen you rewrite the component(s), sharpen these specific boundaries: state the "
        "deciding feature that separates each confused pair, so the two labels stop overlapping."
    )
    return "\n".join(lines)
