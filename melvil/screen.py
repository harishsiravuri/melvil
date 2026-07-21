"""Headroom screening: is this task worth optimizing under this task model?

A user-facing lesson from our own benchmarks: two of six tasks (clinc150,
emotion under gpt-4.1-mini) were effectively solved or ceilinged by the bare
seed prompt, so every optimization dollar spent on them was wasted. `screen()`
answers that question for one dev-eval's worth of budget, before you optimize.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from melvil.artifact import render_prompt
from melvil.config import Config
from melvil.costs import lm_usage, usage_cost_usd
from melvil.data import Example, stratified_sample
from melvil.lmutil import make_lm
from melvil.program import classify_batch

#: verdict bands on seed-prompt accuracy (inclusive lower bounds)
SATURATED_AT = 0.95
MARGINAL_AT = 0.85


@dataclass
class ScreenResult:
    seed_accuracy: float
    n: int
    verdict: str  # "saturated" | "marginal" | "headroom"
    noise_floor: float
    weak_labels: list[str] = field(default_factory=list)  # labels with recall 0 in the sample
    unparseable: int = 0
    cost_usd: float = 0.0
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"seed prompt: {self.seed_accuracy:.3f} accuracy on {self.n} examples "
            f"(±{self.noise_floor:.3f}) -> {self.verdict.upper()}. {self.notes}"
        )


def screen(
    spec_or_artifact,
    data: list[Example],
    config: Config,
    sample_size: int = 100,
    seed: int = 0,
) -> ScreenResult:
    """Evaluate the unoptimized seed prompt on a stratified sample of `data`
    and give a verdict:

    - ``saturated`` (>= 0.95): the seed prompt already solves the task under
      this model; optimization budget will mostly measure noise.
    - ``marginal`` (0.85–0.95): some headroom; expect small gains.
    - ``headroom`` (< 0.85): optimization is worth running.

    Costs one sample-sized evaluation (uses `config.task_model`). The sample is
    drawn from `data` — use your dev split, never test.
    """
    from melvil.artifact import PromptArtifact
    from melvil.config import Features
    from melvil.optimize import seed_candidate_for

    if isinstance(spec_or_artifact, PromptArtifact):
        # post-draft/post-optimize headroom check: evaluate the artifact's
        # rendered prompt (draft -> screen -> maybe optimize workflow)
        artifact = spec_or_artifact
        label_names = artifact.label_names
        rendered = artifact.render()
    else:
        spec = spec_or_artifact
        label_names = spec.label_names
        # blob-mode seed: the same baseline the benchmarks call "seed prompt"
        components = seed_candidate_for(spec, Features.none())
        rendered = render_prompt(components, label_names)
    if len(data) > sample_size:
        idx = stratified_sample(data, sample_size, random.Random(seed))
        sample = [data[i] for i in idx]
    else:
        sample = list(data)

    lm = make_lm(config.task_model, config.task_temperature, config.task_max_tokens)
    since = len(lm.history)
    preds = classify_batch(lm, rendered, sample, label_names, config.num_threads)
    acc = sum(1 for p in preds if p.correct) / len(preds)
    floor = (acc * (1 - acc) / len(preds)) ** 0.5

    weak = []
    for name in label_names:
        support = [p for p in preds if p.gold == name]
        if support and not any(p.correct for p in support):
            weak.append(name)

    if acc >= SATURATED_AT:
        verdict, notes = "saturated", (
            "The seed prompt already solves this task under this model — optimization "
            "will mostly measure noise. Consider a cheaper model or skip optimization."
        )
    elif acc >= MARGINAL_AT:
        verdict, notes = "marginal", "Some headroom; expect small gains from optimization."
    else:
        verdict, notes = "headroom", "Clear headroom; optimization is worth running."
    if weak:
        notes += f" Labels with zero correct predictions in the sample: {weak}."

    return ScreenResult(
        seed_accuracy=round(acc, 4),
        n=len(preds),
        verdict=verdict,
        noise_floor=round(floor, 4),
        weak_labels=weak,
        unparseable=sum(1 for p in preds if not p.predicted),
        cost_usd=round(usage_cost_usd(lm_usage(lm, since), config.prices), 4),
        notes=notes,
    )
