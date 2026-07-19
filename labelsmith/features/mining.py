"""Hard-example mining (feature 3 of the classification layer).

Dev examples that were misclassified persistently across the run become
candidate few-shot exemplars, selected to cover the top confused boundaries of
the final prompt. The exemplar-augmented prompt is kept only if it does not
hurt dev accuracy (checked with one extra dev evaluation, whose budget is
reserved up front).

Caveat (documented in README): exemplars are drawn from dev, so the dev score
of an exemplar-augmented artifact is mildly optimistic; the test score is the
unbiased number.
"""

from __future__ import annotations

from labelsmith.data import Example
from labelsmith.features.confusion import ConfusionState
from labelsmith.program import Prediction


def select_exemplars(
    state: ConfusionState,
    final_preds: list[Prediction],
    per_pair: int = 2,
    max_total: int = 6,
) -> list[dict[str, str]]:
    """Pick exemplars from persistently missed dev examples.

    Selection: for each of the final prompt's top confused (gold, predicted)
    pairs, take the still-misclassified examples with the highest historical
    miss count; label them with their GOLD label. Skips unparseable-output
    pairs (those are format issues, not boundary issues).
    """
    wrong_now = {p.text: p for p in final_preds if not p.correct}
    chosen: list[dict[str, str]] = []
    used: set[str] = set()
    for gold, predicted, _ in state.last_pairs:
        if not predicted:
            continue
        pool = [
            p for p in final_preds
            if not p.correct and p.gold == gold and p.predicted == predicted
            and p.text not in used and p.text in wrong_now
        ]
        pool.sort(key=lambda p: -state.miss_counts.get(p.text, 0))
        for p in pool[:per_pair]:
            chosen.append({"text": p.text, "label": p.gold, "boundary": f"{gold}|{predicted}"})
            used.add(p.text)
            if len(chosen) >= max_total:
                return chosen
    return chosen


def exemplars_as_examples(exemplars: list[dict[str, str]]) -> list[Example]:
    return [Example(text=e["text"], label=e["label"]) for e in exemplars]
