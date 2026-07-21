"""draft(): error-grounded prompt writing — melvil's primary entry point.

One full dev evaluation of the seed prompt -> structured diagnosis (per-label
accuracy, confusion matrix, top confused pairs with concrete misclassified
examples, aggregate error patterns) -> ONE reflection-model call writes the
complete replacement prompt. A second diagnose+rewrite iteration runs by
default (``iterations=2`` — the arm confirmed in our frozen-protocol
benchmarks: 60% (gpt-4.1 family) to 91% (claude-4.5 family) of full GEPA's
gain at roughly a tenth of its optimization cost, beating full GEPA outright
on 2 of 8 tasks; see benchmarks and mechanistic/RESULTS.md).

The procedure and prompts are ported verbatim from the campaign's E8 runner
(mechanistic/e8_diagnose.py) — the code that produced the frozen numbers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from melvil._about import LIBRARY_NAME
from melvil.artifact import BLOB_COMPONENT, OUTPUT_CONTRACT, PromptArtifact, render_prompt
from melvil.config import Config, Features
from melvil.costs import lm_usage, usage_cost_usd
from melvil.data import Example
from melvil.evaluate import accuracy, macro_f1, per_label_stats
from melvil.features.confusion import confusion_matrix, top_confused_pairs
from melvil.lmutil import make_lm, reflection_callable
from melvil.optimize import _validate_data, seed_candidate_for
from melvil.program import classify_batch
from melvil.taskspec import TaskSpec

logger = logging.getLogger(LIBRARY_NAME)

REWRITE_TEMPLATE = """You are an expert prompt engineer. A text classifier runs with the SYSTEM PROMPT below. It was evaluated on {n} held-out examples; a structured error diagnosis follows.

CURRENT SYSTEM PROMPT:
```
{prompt}
```

DIAGNOSIS (accuracy {acc:.1%}):
{diagnosis}

Write a NEW, complete replacement system prompt that fixes the diagnosed failure modes. Requirements:
- It must list every category name exactly as given: {labels}.
- Add whatever category definitions, decision rules, and tie-breakers the errors call for; keep it compact enough to follow reliably.
- Preserve this exact final line, verbatim, as the last line: "{contract}"
- Do not invent categories, do not include few-shot examples.

Return ONLY the new system prompt inside one fenced code block."""


def build_diagnosis(preds, label_names: list[str]) -> str:
    """Human- and LLM-readable diagnosis of one dev evaluation."""
    stats = per_label_stats(preds, label_names)
    lines = ["Per-category recall (worst first):"]
    for name, s in sorted(stats.items(), key=lambda kv: kv[1]["recall"]):
        if s["support"]:
            lines.append(f"- {name}: recall {s['recall']:.2f}, precision {s['precision']:.2f} "
                         f"(n={s['support']})")
    pairs = top_confused_pairs(preds, k=4)
    if pairs:
        lines.append("\nMost frequent confusions (true -> predicted, with examples):")
        for gold, pred_lab, count in pairs:
            shown = [p for p in preds if not p.correct and p.gold == gold
                     and p.predicted == pred_lab][:3]
            lines.append(f"- '{gold}' misread as '{pred_lab or '(unparseable)'}' x{count}:")
            for p in shown:
                lines.append(f"    TEXT: {p.text[:200]}")
    n_unp = sum(1 for p in preds if not p.predicted)
    if n_unp:
        lines.append(f"\n{n_unp} responses failed to parse as a category name.")
    return "\n".join(lines)


def _extract_prompt(response: str) -> str:
    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    prompt = max(blocks, key=len).strip() if blocks else response.strip()
    if OUTPUT_CONTRACT not in prompt:
        prompt = prompt.rstrip() + "\n\n" + OUTPUT_CONTRACT
    return prompt


def draft(
    spec: TaskSpec,
    train: list[Example],
    dev: list[Example],
    config: Config,
    *,
    iterations: int = 2,
    resume: bool = False,
    parent: PromptArtifact | None = None,
) -> PromptArtifact:
    """Write an optimized classifier prompt from an error diagnosis — no
    evolutionary search. Cost: ``iterations`` dev evaluations + reflection
    calls (roughly a tenth of a light-budget ``optimize()`` run).

    The recommended workflow is draft -> ``screen(artifact, ...)`` -> escalate
    to ``optimize()`` only if meaningful headroom remains and the extra
    accuracy is worth ~10x the spend.

    ``train`` is accepted for signature parity and label validation but the
    procedure only uses ``dev`` (as in the benchmarked arm). The run directory
    receives diagnosis_x{i}.txt files and artifact.json.
    """
    _validate_data(spec, train, dev)
    run_dir = Path(config.run_dir) if config.run_dir else (
        Path("runs") / spec.name / f"draft-{config.config_hash}-s{config.seed}")
    artifact_path = run_dir / "artifact.json"
    if artifact_path.exists():
        if resume:
            return PromptArtifact.load(artifact_path)
        raise FileExistsError(
            f"{artifact_path} exists. Pass resume=True to reuse it, or set a fresh "
            "Config.run_dir.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=1, default=str))

    task_lm = make_lm(config.task_model, config.task_temperature, config.task_max_tokens)
    refl_lm = make_lm(config.reflection_model, config.reflection_temperature,
                      config.reflection_max_tokens, rollout_id=config.seed)
    refl_call = reflection_callable(refl_lm)
    t0, r0 = len(task_lm.history), len(refl_lm.history)

    seed_components = seed_candidate_for(spec, Features.none())
    current = render_prompt(seed_components, spec.label_names)
    seed_dev_acc: float | None = None
    diagnoses: list[str] = []
    dev_accs: list[float] = []

    for it in range(1, iterations + 1):
        preds = classify_batch(task_lm, current, dev, spec.label_names, config.num_threads)
        acc = accuracy(preds)
        if seed_dev_acc is None:
            seed_dev_acc = acc
        diagnosis = build_diagnosis(preds, spec.label_names)
        diagnoses.append(diagnosis)
        dev_accs.append(acc)
        (run_dir / f"diagnosis_x{it}.txt").write_text(
            f"input prompt dev accuracy: {acc:.4f}\n\n{diagnosis}")
        response = refl_call(REWRITE_TEMPLATE.format(
            n=len(dev), prompt=current, acc=acc, diagnosis=diagnosis,
            labels=", ".join(spec.label_names), contract=OUTPUT_CONTRACT))
        current = _extract_prompt(response)
        logger.info("draft iteration %d: input dev %.3f -> rewrote prompt", it, acc)

    # score the final draft on dev (cache-cheap where texts repeat)
    final_preds = classify_batch(task_lm, current, dev, spec.label_names, config.num_threads)
    final_acc = accuracy(final_preds)
    stats = per_label_stats(final_preds, spec.label_names)
    usage_t, usage_r = lm_usage(task_lm, t0), lm_usage(refl_lm, r0)

    blob_stored = current
    while blob_stored.rstrip().endswith(OUTPUT_CONTRACT):
        blob_stored = blob_stored.rstrip()[: -len(OUTPUT_CONTRACT)].rstrip()
    artifact = PromptArtifact(
        task_name=spec.name,
        components={BLOB_COMPONENT: blob_stored},
        label_names=spec.label_names,
        models={"task": config.task_model, "reflection": config.reflection_model},
        budget={
            "method": "draft",
            "iterations": iterations,
            "metric_calls_spent": (iterations + 1) * len(dev),
            "cost_usd": round(usage_cost_usd(usage_t, config.prices)
                              + usage_cost_usd(usage_r, config.prices), 4),
            "task_lm_usage": usage_t, "reflection_lm_usage": usage_r,
        },
        scores={
            "dev_accuracy": final_acc,
            "dev_macro_f1": macro_f1(stats),
            "per_label_dev": stats,
            "seed_dev_accuracy": seed_dev_acc,
            "dev_accuracy_per_iteration": dev_accs + [final_acc],
        },
        confusion=confusion_matrix(final_preds, spec.label_names),
        config=config.to_dict(),
        config_hash=config.config_hash,
        parent_id=parent.artifact_id if parent else None,
        notes=f"draft x{iterations} (error-grounded rewrite; no evolutionary search)",
    )
    # store the diagnoses inside the artifact for auditability
    artifact.config["draft_diagnoses"] = diagnoses
    artifact.save(artifact_path)
    logger.info("draft done: dev %.3f (seed %.3f) | $%.2f | artifact %s",
                final_acc, seed_dev_acc, artifact.budget["cost_usd"], artifact.artifact_id)
    return artifact
