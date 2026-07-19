"""Hard-example mining (feature 3 of the classification layer).

Dev examples that were misclassified persistently across the run become
candidate few-shot exemplars, selected to cover the top confused boundaries of
the final prompt. Since v0.2, three protections (each motivated by v0.1
benchmark findings) guard against the classic trap that persistently-missed
examples select for label noise:

1. consistency prescreen (free): an example missed in every round with the
   same wrong prediction every time looks like label noise, not a boundary
   case — excluded here;
2. optional LLM screen (one reflection-model call): candidate exemplars whose
   gold label contradicts the taxonomy's own definitions are dropped;
3. quarantined accept gate (in `optimize`): the exemplar-augmented prompt must
   beat the base prompt by more than the binomial noise floor on the dev
   examples EXCLUDING the exemplars themselves.

Caveat still documented in README: exemplars come from dev; the test score is
the unbiased number.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from melvil.data import Example
from melvil.features.confusion import ConfusionState
from melvil.program import Prediction

#: prescreen thresholds: missed in >=90% of rounds AND >=90% of misses were the
#: same wrong prediction -> suspected label noise
NOISE_MISS_RATE = 0.9
NOISE_CONSISTENCY = 0.9


def looks_like_label_noise(state: ConfusionState, text: str) -> bool:
    """True when the model never wavered: wrong in (almost) every round, always
    the same way. Boundary-hard-but-learnable examples get fixed by at least
    some candidate prompts, so their miss rate is lower or their errors vary.
    (Heuristic — a consistently-confused boundary case can trip it; the
    quarantined gate is the backstop.)"""
    misses = state.miss_counts.get(text, 0)
    if state.rounds < 3 or misses < NOISE_MISS_RATE * state.rounds:
        return False
    pair_counts = state.miss_pairs.get(text)
    if not pair_counts:
        return False
    return max(pair_counts.values()) >= NOISE_CONSISTENCY * misses


def select_exemplars(
    state: ConfusionState,
    final_preds: list[Prediction],
    per_pair: int = 2,
    max_total: int = 6,
) -> list[dict[str, str]]:
    """Pick exemplars from persistently missed dev examples.

    Selection: for each of the final prompt's top confused (gold, predicted)
    pairs, take the still-misclassified examples with the highest historical
    miss count that pass the label-noise prescreen; label them with their GOLD
    label. Skips unparseable-output pairs (format issues, not boundary issues).
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
            and not looks_like_label_noise(state, p.text)
        ]
        pool.sort(key=lambda p: -state.miss_counts.get(p.text, 0))
        for p in pool[:per_pair]:
            chosen.append({"text": p.text, "label": p.gold, "boundary": f"{gold}|{predicted}"})
            used.add(p.text)
            if len(chosen) >= max_total:
                return chosen
    return chosen


def llm_screen(
    exemplars: list[dict[str, str]],
    label_definitions: dict[str, str],
    call_lm: Callable[[str], str],
) -> list[dict[str, str]]:
    """Drop candidate exemplars whose gold label contradicts the taxonomy's own
    definitions, judged by the reflection model in ONE call. On any parse
    failure the exemplar is kept (screen is advisory, the gate is the
    backstop)."""
    if not exemplars:
        return exemplars
    defs = "\n".join(f"- {name}: {desc}" for name, desc in label_definitions.items() if desc)
    items = "\n".join(
        f"{i}. TEXT: {e['text'][:300]}\n   ASSIGNED LABEL: {e['label']}"
        for i, e in enumerate(exemplars)
    )
    prompt = (
        "You are auditing a labeled dataset for label noise. Category definitions:\n"
        f"{defs or '(names only, no definitions)'}\n\n"
        "For each numbered item, decide whether the ASSIGNED LABEL is plausible for the "
        "TEXT under these definitions. Respond with ONLY a JSON array of the item numbers "
        f"whose label is plausible (e.g. [0, 2]).\n\n{items}"
    )
    try:
        out = call_lm(prompt)
        m = re.search(r"\[[\d,\s]*\]", out)
        if not m:
            return exemplars
        keep_idx = set(json.loads(m.group(0)))
        return [e for i, e in enumerate(exemplars) if i in keep_idx]
    except Exception:  # noqa: BLE001 - advisory screen must never kill the run
        return exemplars


def exemplars_as_examples(exemplars: list[dict[str, str]]) -> list[Example]:
    return [Example(text=e["text"], label=e["label"]) for e in exemplars]
