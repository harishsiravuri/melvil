"""The classifier at runtime: rendered prompt -> messages -> completion -> label.

The system prompt used here is exactly `render_prompt(...)` — what
`PromptArtifact.render()` returns — so the deployable artifact is the very
prompt that was optimized and evaluated.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from melvil.data import Example, canon


@dataclass
class Prediction:
    text: str
    gold: str
    predicted: str  # parsed label name, or "" if unparseable
    raw: str
    correct: bool
    feedback: str


def build_messages(rendered_prompt: str, text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": rendered_prompt},
        {"role": "user", "content": text},
    ]


def parse_label(raw: str, label_names: list[str]) -> str:
    """Parse a completion into a label name. Tries: full completion, then the
    last non-empty line, then a unique canonical substring match. Returns ""
    when nothing matches."""
    by_canon = {canon(name): name for name in label_names}
    whole = canon(raw)
    if whole in by_canon:
        return by_canon[whole]
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines:
        last = canon(lines[-1])
        if last in by_canon:
            return by_canon[last]
    hits = [name for c, name in by_canon.items() if c and c in whole]
    if len(hits) == 1:
        return hits[0]
    return ""


def feedback_for(gold: str, predicted: str, raw: str, label_names: list[str]) -> tuple[float, str]:
    """(score, feedback) for one example — feedback phrasing salvaged from the
    pilot; it is what the reflection LM reads."""
    if predicted and canon(predicted) == canon(gold):
        return 1.0, f"Correct: '{gold}' is the right category."
    if not predicted:
        return 0.0, (
            f"Incorrect: the response {raw[:120]!r} could not be parsed as one of the "
            f"allowed categories. The correct label is '{gold}'."
        )
    return 0.0, f"Incorrect: the model predicted '{predicted}' but the correct label is '{gold}'."


def classify_batch(
    lm: Any,
    rendered_prompt: str,
    examples: list[Example],
    label_names: list[str],
    num_threads: int = 8,
) -> list[Prediction]:
    """Run the classifier over a batch. Individual failures score 0 with an
    error note (never raises for one example)."""

    def run(ex: Example) -> Prediction:
        try:
            out = lm(messages=build_messages(rendered_prompt, ex.text))
            raw = out[0] if isinstance(out, list) else out
            if isinstance(raw, dict):
                raw = raw.get("text", "")
            raw = str(raw).strip()
        except Exception as e:  # noqa: BLE001 - per-example errors must not kill the run
            raw = f"<lm error: {e}>"
        predicted = parse_label(raw, label_names)
        score, fb = feedback_for(ex.label, predicted, raw, label_names)
        return Prediction(
            text=ex.text, gold=ex.label, predicted=predicted, raw=raw,
            correct=score >= 1.0, feedback=fb,
        )

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(run, examples))
