"""Evaluation: accuracy, macro-F1, per-label precision/recall, confusion, cost.

`evaluate(artifact, data, model=...)` also serves as transfer evaluation: pass
a different model than the one the artifact was optimized on; the report
records both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from melvil.artifact import PromptArtifact
from melvil.costs import lm_usage, usage_cost_usd
from melvil.data import Example
from melvil.features.confusion import confusion_matrix, top_confused_pairs
from melvil.lmutil import make_lm
from melvil.program import Prediction, classify_batch


def per_label_stats(preds: list[Prediction], label_names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in label_names:
        tp = sum(1 for p in preds if p.predicted == name and p.gold == name)
        fp = sum(1 for p in preds if p.predicted == name and p.gold != name)
        fn = sum(1 for p in preds if p.gold == name and p.predicted != name)
        support = sum(1 for p in preds if p.gold == name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return stats


def macro_f1(stats: dict[str, dict[str, float]], skip_empty: bool = True) -> float:
    rows = [s for s in stats.values() if s["support"] > 0] if skip_empty else list(stats.values())
    return round(sum(s["f1"] for s in rows) / len(rows), 4) if rows else 0.0


def accuracy(preds: list[Prediction]) -> float:
    return round(sum(1 for p in preds if p.correct) / len(preds), 4) if preds else 0.0


@dataclass
class Report:
    artifact_id: str
    task_name: str
    model: str
    optimized_on_model: str
    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[str, dict[str, float]]
    confusion: dict[str, Any]
    top_confusions: list[tuple[str, str, int]]
    cost_usd: float
    unparseable: int
    transfer: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["top_confusions"] = [list(t) for t in self.top_confusions]
        return d


def evaluate(
    artifact: PromptArtifact,
    data: list[Example],
    model: str | None = None,
    num_threads: int = 8,
    prices: dict[str, tuple[float, float]] | None = None,
) -> Report:
    """Run the artifact's rendered prompt over `data` and score it. `model`
    defaults to the model the artifact was optimized on; passing a different
    one is a transfer evaluation."""
    opt_model = artifact.models.get("task", "")
    model = model or opt_model
    if not model:
        raise ValueError("no model given and the artifact records none")
    lm = make_lm(model, temperature=0.0, max_tokens=artifact.config.get("task_max_tokens", 40))
    since = len(lm.history)
    preds = classify_batch(lm, artifact.render(), data, artifact.label_names, num_threads)
    usage = lm_usage(lm, since=since)
    stats = per_label_stats(preds, artifact.label_names)
    return Report(
        artifact_id=artifact.artifact_id,
        task_name=artifact.task_name,
        model=model,
        optimized_on_model=opt_model,
        n=len(preds),
        accuracy=accuracy(preds),
        macro_f1=macro_f1(stats),
        per_label=stats,
        confusion=confusion_matrix(preds, artifact.label_names),
        top_confusions=top_confused_pairs(preds, k=5),
        cost_usd=round(usage_cost_usd(usage, prices), 4),
        unparseable=sum(1 for p in preds if not p.predicted),
        transfer=bool(opt_model and model != opt_model),
    )


def report(obj: Report | PromptArtifact) -> str:
    """Markdown rendering of a Report (or of an artifact's stored scores) —
    printable in notebooks, pasteable into docs."""
    if isinstance(obj, PromptArtifact):
        return _artifact_report(obj)
    r = obj
    lines = [
        f"## Evaluation — {r.task_name} (`{r.artifact_id}`)",
        "",
        f"- model: `{r.model}`"
        + (f" (TRANSFER — optimized on `{r.optimized_on_model}`)" if r.transfer else ""),
        f"- examples: {r.n} | accuracy: **{r.accuracy:.3f}** | macro-F1: **{r.macro_f1:.3f}**"
        f" | unparseable outputs: {r.unparseable} | cost: ${r.cost_usd:.2f}",
        "",
        "| label | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for name, s in r.per_label.items():
        lines.append(
            f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} "
            f"| {s['f1']:.3f} | {s['support']} |"
        )
    if r.top_confusions:
        lines += ["", "Top confusions (gold → predicted × count):", ""]
        for gold, pred, c in r.top_confusions:
            lines.append(f"- {gold} → {pred or '(unparseable)'} × {c}")
    return "\n".join(lines)


def _artifact_report(a: PromptArtifact) -> str:
    s = a.scores
    lines = [
        f"## Artifact `{a.artifact_id}` — {a.task_name}",
        "",
        f"- created: {a.created_at} | parent: {a.parent_id or '—'} | config: `{a.config_hash}`",
        f"- models: task `{a.models.get('task', '?')}`, "
        f"reflection `{a.models.get('reflection', '?')}`",
        f"- budget: {a.budget.get('metric_calls_spent', '?')} metric calls,"
        f" ${a.budget.get('cost_usd', 0):.2f}",
        f"- dev accuracy: **{s.get('dev_accuracy', float('nan')):.3f}**"
        f" | dev macro-F1: **{s.get('dev_macro_f1', float('nan')):.3f}**"
        + (f" | test accuracy: **{s['test_accuracy']:.3f}**" if "test_accuracy" in s else ""),
        f"- components: {len(a.components)} | exemplars: {len(a.exemplars)}",
    ]
    if a.curve:
        last = a.curve[-1]
        lines.append(
            f"- optimization: {len(a.curve)} candidates, best found at "
            f"{last.get('metric_calls', '?')} metric calls"
        )
    return "\n".join(lines)
