"""Coarse-to-fine optimization (campaign E4 / claim C4).

Motivated by the confirmation-pass finding: whole-prompt rewrites discover
global strategies; per-component edits fragment a small budget. So sequence
them — phase 1 evolves a single free-text prompt (vanilla GEPA); once progress
stalls (`s` consecutive rejected proposals, or a fixed budget fraction), the
winning prompt is DECOMPOSED by the reflection model into codebook components
(task instruction + per-label definitions + boundary rules), verified by a
cache-cheap dev eval, and phase 2 refines components under the
confusion-driven selector with the remaining budget.

Entry point: ``optimize_c2f(spec, train, dev, config, switch=...)`` where
switch is ``("rejections", 2)`` or ``("fraction", 0.6)``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from melvil._about import LIBRARY_NAME
from melvil.adapter import ClassificationAdapter
from melvil.artifact import (
    BLOB_COMPONENT,
    BOUNDARY_COMPONENT,
    TASK_COMPONENT,
    PromptArtifact,
    label_component,
    render_prompt,
)
from melvil.config import Config, Features
from melvil.costs import lm_usage, usage_cost_usd
from melvil.data import Example, canon
from melvil.evaluate import accuracy, macro_f1, per_label_stats
from melvil.features.confusion import ConfusionComponentSelector, confusion_matrix
from melvil.lmutil import make_lm, reflection_callable
from melvil.optimize import seed_candidate_for
from melvil.program import classify_batch
from melvil.taskspec import TaskSpec

logger = logging.getLogger(LIBRARY_NAME)

DECOMPOSE_TEMPLATE = """Split the following classification system prompt into named components, preserving ALL of its content and meaning (rephrase as little as possible; distribute every rule to the most fitting component).

PROMPT:
```
{prompt}
```

Return one fenced block per component, exactly in this format, one block per category plus the two shared blocks:

```component: task_instruction
<general task guidance, NOT category-specific>
```
```component: label::<category name>
<everything the prompt says about that category>
```
```component: boundary_rules
<cross-category tie-breakers and disambiguation rules>
```

The categories are: {labels}. Every category must get a block (empty is allowed). Do not add new rules."""

_BLOCK_RE = re.compile(r"```component:\s*(?P<name>[^\n`]+)\n(?P<body>.*?)```", re.DOTALL)


def decompose_prompt(blob: str, spec: TaskSpec, refl_call) -> dict[str, str]:
    """LLM-split a blob prompt into codebook components; fall back to seeding
    the codebook with the blob as task_instruction on parse failure."""
    out = refl_call(DECOMPOSE_TEMPLATE.format(prompt=blob, labels=", ".join(spec.label_names)))
    components: dict[str, str] = {}
    valid = {TASK_COMPONENT, BOUNDARY_COMPONENT} | {
        label_component(n) for n in spec.label_names}
    by_canon = {canon(n): n for n in spec.label_names}
    for m in _BLOCK_RE.finditer(out):
        name = m.group("name").strip()
        if name.startswith("label::"):
            mapped = by_canon.get(canon(name.split("::", 1)[1]))
            name = label_component(mapped) if mapped else name
        if name in valid:
            components[name] = m.group("body").strip()
    if TASK_COMPONENT not in components:
        components[TASK_COMPONENT] = blob  # degenerate fallback: keep everything
    for n in spec.label_names:
        components.setdefault(label_component(n), "")
    components.setdefault(BOUNDARY_COMPONENT, "")
    return components


def optimize_c2f(
    spec: TaskSpec,
    train: list[Example],
    dev: list[Example],
    config: Config,
    *,
    switch: tuple[str, float] = ("rejections", 2),
    resume: bool = False,
) -> PromptArtifact:
    """Two-phase coarse-to-fine run. Budget = config.metric_calls total across
    both phases (decomposition-verification dev eval is cache-cheap but counted)."""
    import gepa

    run_dir = Path(config.run_dir) if config.run_dir else Path("runs") / spec.name / "c2f"
    artifact_path = run_dir / "artifact.json"
    if artifact_path.exists() and resume:
        return PromptArtifact.load(artifact_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    fam_task = make_lm(config.task_model, config.task_temperature, config.task_max_tokens)
    refl_lm = make_lm(config.reflection_model, config.reflection_temperature,
                      config.reflection_max_tokens, rollout_id=config.seed)
    refl_call = reflection_callable(refl_lm)
    t0, r0 = len(fam_task.history), len(refl_lm.history)

    class _Quiet:
        def log(self, message: str) -> None:
            logger.debug("gepa: %s", message)

    total_budget = config.metric_calls
    mode, s_val = switch
    phase1_cap = int(total_budget * s_val) if mode == "fraction" else total_budget

    # ---------------------------------------------------------------- phase 1
    adapter1 = ClassificationAdapter(
        spec, fam_task, config, valset=dev, trace_path=run_dir / "traces_p1.jsonl")
    proposals = {"n": 0, "at_last_accept": 0}
    rounds_seen = {"n": 0}

    def refl_counting(prompt: str) -> str:
        proposals["n"] += 1
        return refl_call(prompt)

    def stalled(gepa_state) -> bool:
        if mode != "rejections":
            return False
        if adapter1._round > rounds_seen["n"]:  # a full-val eval => an accept happened
            rounds_seen["n"] = adapter1._round
            proposals["at_last_accept"] = proposals["n"]
        # ignore the pre-first-accept phase (round 1 is the seed's full eval)
        return (adapter1._round >= 2
                and proposals["n"] - proposals["at_last_accept"] >= s_val)

    seed_cand = seed_candidate_for(spec, Features.none())
    r1 = gepa.optimize(
        seed_candidate=seed_cand, trainset=train, valset=dev, adapter=adapter1,
        reflection_lm=refl_counting, max_metric_calls=phase1_cap,
        run_dir=str(run_dir / "p1"), seed=config.seed, logger=_Quiet(),
        stop_callbacks=[stalled] if mode == "rejections" else None,
        display_progress_bar=False,
    )
    s1 = list(r1.val_aggregate_scores)
    best1_idx = max(range(len(s1)), key=lambda i: s1[i])
    best_blob = r1.candidates[best1_idx][BLOB_COMPONENT]
    spent1 = adapter1._metric_calls
    logger.info("c2f phase1: dev %.3f after %d calls (%d proposals)",
                s1[best1_idx], spent1, proposals["n"])

    remaining = total_budget - spent1 - len(dev)  # reserve decomposition check
    curve1 = [{"metric_calls": int(r1.discovery_eval_counts[i]),
               "dev_score": round(s1[i], 4), "phase": 1} for i in range(len(s1))]

    # ------------------------------------------------- decompose + verify
    components = decompose_prompt(best_blob, spec, refl_call)
    dec_preds = classify_batch(fam_task, render_prompt(components, spec.label_names),
                               dev, spec.label_names, config.num_threads)
    dec_acc = accuracy(dec_preds)
    blob_acc = s1[best1_idx]
    (run_dir / "decomposition.json").write_text(json.dumps(
        {"blob_dev": blob_acc, "decomposed_dev": dec_acc, "components": components}, indent=1))
    floor = (blob_acc * (1 - blob_acc) / max(1, len(dev))) ** 0.5
    if dec_acc < blob_acc - 2 * floor or remaining < 2 * len(dev):
        # decomposition lost meaning or nothing left to refine: return phase-1 result
        logger.info("c2f: skipping phase 2 (decomposed dev %.3f vs blob %.3f, remaining %d)",
                    dec_acc, blob_acc, max(0, remaining))
        final_components = {BLOB_COMPONENT: best_blob}
        final_preds = classify_batch(
            fam_task, render_prompt(final_components, spec.label_names), dev,
            spec.label_names, config.num_threads)
        curve = curve1
        phase2_note = f"skipped (decomposed dev {dec_acc:.3f} vs blob {blob_acc:.3f})"
    else:
        # ------------------------------------------------------------ phase 2
        import dataclasses

        feats2 = Features(codebook=True, confusion_reflection=True, hard_example_mining=False)
        cfg2 = dataclasses.replace(config, features=feats2, budget=remaining)
        adapter2 = ClassificationAdapter(
            spec, fam_task, cfg2, valset=dev, trace_path=run_dir / "traces_p2.jsonl")
        if adapter2.proposer is not None:
            adapter2.proposer.bind_lm(refl_call)
        selector = ConfusionComponentSelector(adapter2.confusion, list(components))
        r2 = gepa.optimize(
            seed_candidate=components, trainset=train, valset=dev, adapter=adapter2,
            reflection_lm=refl_call, module_selector=selector,
            max_metric_calls=remaining, run_dir=str(run_dir / "p2"),
            seed=config.seed, logger=_Quiet(), display_progress_bar=False,
        )
        s2 = list(r2.val_aggregate_scores)
        best2_idx = max(range(len(s2)), key=lambda i: s2[i])
        curve2 = [{"metric_calls": spent1 + len(dev) + int(r2.discovery_eval_counts[i]),
                   "dev_score": round(s2[i], 4), "phase": 2} for i in range(len(s2))]
        curve = curve1 + curve2
        if s2[best2_idx] >= blob_acc:
            final_components = dict(r2.candidates[best2_idx])
            phase2_note = f"phase2 won (dev {s2[best2_idx]:.3f} >= {blob_acc:.3f})"
        else:
            final_components = {BLOB_COMPONENT: best_blob}
            phase2_note = f"phase1 blob kept (phase2 best {s2[best2_idx]:.3f} < {blob_acc:.3f})"
        final_preds = classify_batch(
            fam_task, render_prompt(final_components, spec.label_names), dev,
            spec.label_names, config.num_threads)

    stats = per_label_stats(final_preds, spec.label_names)
    artifact = PromptArtifact(
        task_name=spec.name, components=final_components, label_names=spec.label_names,
        models={"task": config.task_model, "reflection": config.reflection_model},
        budget={"metric_calls_budget": total_budget,
                "cost_usd": round(usage_cost_usd(lm_usage(fam_task, t0))
                                  + usage_cost_usd(lm_usage(refl_lm, r0)), 4)},
        scores={"dev_accuracy": accuracy(final_preds), "dev_macro_f1": macro_f1(stats),
                "seed_dev_accuracy": round(s1[0], 4),
                "phase1_dev": round(blob_acc, 4)},
        confusion=confusion_matrix(final_preds, spec.label_names),
        curve=curve, config=config.to_dict(), config_hash=config.config_hash,
        notes=f"c2f switch={switch}; {phase2_note}",
    )
    artifact.save(artifact_path)
    return artifact
