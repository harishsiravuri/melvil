"""`optimize(taskspec, train, dev, config) -> PromptArtifact` — the core entry
point. Wires the classification adapter, features, and LMs into the gepa
engine; manages run directories, resume, callbacks, and cost accounting."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from labelsmith._about import LIBRARY_NAME
from labelsmith.adapter import ClassificationAdapter, RoundInfo
from labelsmith.artifact import (
    BLOB_COMPONENT,
    BOUNDARY_COMPONENT,
    TASK_COMPONENT,
    PromptArtifact,
    label_component,
    render_prompt,
)
from labelsmith.config import Config
from labelsmith.costs import lm_usage, usage_cost_usd
from labelsmith.data import Example, canon
from labelsmith.evaluate import accuracy, macro_f1, per_label_stats
from labelsmith.features.confusion import ConfusionComponentSelector, confusion_matrix
from labelsmith.features.mining import select_exemplars
from labelsmith.lmutil import make_lm, reflection_callable
from labelsmith.program import classify_batch
from labelsmith.taskspec import TaskSpec

logger = logging.getLogger(LIBRARY_NAME)

DEFAULT_TASK_INSTRUCTION = (
    "You are a precise text classifier. Read the input text and assign it to exactly one "
    "of the categories defined below."
)


def seed_candidate_for(spec: TaskSpec, features=None) -> dict[str, str]:
    """The initial candidate. Codebook mode: one component per label plus task
    instruction and boundary rules. Blob mode: a single free-text instruction
    (vanilla GEPA)."""
    codebook = features.codebook if features is not None else True
    if codebook:
        cand = {TASK_COMPONENT: spec.instruction or DEFAULT_TASK_INSTRUCTION}
        for lb in spec.labels:
            cand[label_component(lb.name)] = lb.description
        notes = [f"{lb.name}: {lb.boundary_notes}" for lb in spec.labels if lb.boundary_notes]
        cand[BOUNDARY_COMPONENT] = "\n".join(notes)
        return cand
    lines = [
        spec.instruction
        or "Classify the given text into one of these categories: "
        + ", ".join(spec.label_names)
        + "."
    ]
    described = [lb for lb in spec.labels if lb.description]
    if described:
        lines.append("Category definitions:")
        lines.extend(f"- {lb.name}: {lb.description}" for lb in described)
    return {BLOB_COMPONENT: "\n".join(lines)}


def _validate_data(spec: TaskSpec, train: list[Example], dev: list[Example]) -> None:
    known = {canon(n) for n in spec.label_names}
    for split_name, split in (("train", train), ("dev", dev)):
        if not split:
            raise ValueError(f"{split_name} set is empty")
        unknown = sorted({e.label for e in split if canon(e.label) not in known})
        if unknown:
            raise ValueError(
                f"{split_name} set contains labels not in the TaskSpec: {unknown[:10]}"
            )


def _default_run_dir(spec: TaskSpec, config: Config) -> Path:
    return Path("runs") / spec.name / f"{config.config_hash}-s{config.seed}"


def optimize(
    spec: TaskSpec,
    train: list[Example],
    dev: list[Example],
    config: Config,
    *,
    resume: bool = False,
    on_round: Callable[[RoundInfo], None] | None = None,
    parent: PromptArtifact | None = None,
) -> PromptArtifact:
    """Optimize a classifier prompt for `spec` and return a PromptArtifact.

    The run directory (default ``runs/<task>/<confighash>-s<seed>``) receives
    config.json, the gepa engine state (which makes ``resume=True`` work after
    an interruption), reflection traces, run.log, and artifact.json. The dev
    set drives all optimization decisions; no other data is touched.
    """
    _validate_data(spec, train, dev)
    run_dir = Path(config.run_dir) if config.run_dir else _default_run_dir(spec, config)
    artifact_path = run_dir / "artifact.json"
    if artifact_path.exists():
        if resume:
            logger.info("run already complete, loading %s", artifact_path)
            return PromptArtifact.load(artifact_path)
        raise FileExistsError(
            f"{artifact_path} exists. Pass resume=True to reuse it, or set a fresh "
            "Config.run_dir to re-run."
        )
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"{run_dir} contains a partial run. Pass resume=True to continue it, or set a "
            "fresh Config.run_dir."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=1, default=str))

    task_lm = make_lm(
        config.task_model, config.task_temperature, config.task_max_tokens
    )
    reflection_lm = make_lm(
        config.reflection_model,
        config.reflection_temperature,
        config.reflection_max_tokens,
        rollout_id=config.seed,
    )
    task_since, refl_since = len(task_lm.history), len(reflection_lm.history)

    def spend() -> float:
        return round(
            usage_cost_usd(lm_usage(task_lm, task_since), config.prices)
            + usage_cost_usd(lm_usage(reflection_lm, refl_since), config.prices),
            4,
        )

    def _log_round(info: RoundInfo) -> None:
        logger.info(
            "round %d: dev %.3f (best %.3f) | %d metric calls | $%.2f",
            info.round, info.dev_score, info.best_dev_score,
            info.metric_calls_spent, info.cost_usd,
        )
        if on_round:
            on_round(info)

    adapter = ClassificationAdapter(
        spec, task_lm, config, valset=dev,
        trace_path=run_dir / "traces.jsonl",
        on_round=_log_round, cost_fn=spend,
    )
    if adapter.proposer is not None:
        adapter.proposer.bind_lm(reflection_callable(reflection_lm))

    feats = config.features
    mining_reserve = len(dev) if feats.hard_example_mining else 0
    gepa_budget = config.metric_calls - mining_reserve
    if gepa_budget < 2 * len(dev):
        logger.warning(
            "budget %d leaves only %d metric calls for optimization (dev=%d); "
            "consider a bigger budget or a smaller dev set",
            config.metric_calls, gepa_budget, len(dev),
        )

    seed_candidate = seed_candidate_for(spec, feats)
    if feats.codebook and feats.confusion_reflection:
        module_selector = ConfusionComponentSelector(adapter.confusion, list(seed_candidate))
    else:
        module_selector = "round_robin"

    import gepa

    class _QuietLogger:
        """Route gepa's chatty per-iteration logging to DEBUG; labelsmith's own
        round logging is the user-facing progress channel."""

        def log(self, message: str) -> None:
            logger.debug("gepa: %s", message)

    t0 = time.time()
    result = gepa.optimize(
        logger=_QuietLogger(),
        seed_candidate=seed_candidate,
        trainset=train,
        valset=dev,
        adapter=adapter,
        reflection_lm=reflection_callable(reflection_lm),
        module_selector=module_selector,
        reflection_minibatch_size=config.reflection_minibatch_size,
        max_metric_calls=gepa_budget,
        run_dir=str(run_dir),
        seed=config.seed,
        display_progress_bar=False,
    )
    opt_seconds = time.time() - t0

    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_candidate = dict(result.candidates[best_idx])
    curve = [
        {"metric_calls": int(result.discovery_eval_counts[i]), "dev_score": round(scores[i], 4)}
        for i in range(len(scores))
    ]

    # Final dev pass of the best candidate (cache-served: gepa already evaluated
    # this exact prompt on dev) -> per-label stats + confusion for the artifact.
    best_rendered = render_prompt(best_candidate, spec.label_names)
    final_preds = classify_batch(task_lm, best_rendered, dev, spec.label_names, config.num_threads)
    dev_acc = accuracy(final_preds)
    stats = per_label_stats(final_preds, spec.label_names)

    # ---------------------------------------------- hard-example mining pass
    exemplars: list[dict[str, str]] = []
    mining_note = ""
    mining_calls = 0
    if feats.hard_example_mining:
        adapter.confusion.update(final_preds)  # ensure state reflects best candidate
        picked = select_exemplars(adapter.confusion, final_preds, max_total=config.max_exemplars)
        if picked:
            aug_rendered = render_prompt(best_candidate, spec.label_names, picked)
            aug_preds = classify_batch(
                task_lm, aug_rendered, dev, spec.label_names, config.num_threads
            )
            mining_calls = len(dev)
            aug_acc = accuracy(aug_preds)
            if aug_acc >= dev_acc:
                mining_note = f"kept {len(picked)} exemplars (dev {aug_acc:.3f} >= {dev_acc:.3f})"
                exemplars = picked
                final_preds = aug_preds
                dev_acc = aug_acc
                stats = per_label_stats(aug_preds, spec.label_names)
            else:
                mining_note = (
                    f"rejected {len(picked)} exemplars (dev {aug_acc:.3f} < {dev_acc:.3f})"
                )
            logger.info("hard-example mining: %s", mining_note)
        else:
            mining_note = "no persistent hard examples found"

    task_usage = lm_usage(task_lm, task_since)
    refl_usage = lm_usage(reflection_lm, refl_since)
    artifact = PromptArtifact(
        task_name=spec.name,
        components=best_candidate,
        label_names=spec.label_names,
        exemplars=exemplars,
        models={"task": config.task_model, "reflection": config.reflection_model},
        budget={
            "metric_calls_budget": config.metric_calls,
            "metric_calls_spent": int(result.total_metric_calls or 0) + mining_calls,
            "cost_usd": spend(),
            "task_lm_usage": task_usage,
            "reflection_lm_usage": refl_usage,
            "opt_seconds": round(opt_seconds, 1),
        },
        scores={
            "dev_accuracy": dev_acc,
            "dev_macro_f1": macro_f1(stats),
            "per_label_dev": stats,
            "seed_dev_accuracy": round(scores[0], 4) if scores else None,
        },
        confusion=confusion_matrix(final_preds, spec.label_names),
        curve=curve,
        config=config.to_dict(),
        config_hash=config.config_hash,
        parent_id=parent.artifact_id if parent else None,
        notes=mining_note,
    )
    artifact.save(artifact_path)
    logger.info(
        "done: dev %.3f (seed %.3f) | %s candidates | $%.2f | artifact %s",
        dev_acc, scores[0] if scores else float("nan"), len(scores), spend(), artifact.artifact_id,
    )
    return artifact
