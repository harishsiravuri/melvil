"""The gepa `GEPAAdapter` for classification: builds the prompt from candidate
components, evaluates batches with the task LM, maintains confusion state on
full dev evaluations, and produces reflective datasets per component."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch

from melvil.artifact import BLOB_COMPONENT, LABEL_PREFIX, render_prompt
from melvil.config import Config
from melvil.data import Example
from melvil.features.confusion import ConfusionState, render_confusion_block
from melvil.program import Prediction, classify_batch
from melvil.proposer import ClassificationProposer
from melvil.taskspec import TaskSpec


@dataclass
class RoundInfo:
    """Passed to the `on_round` callback after every full dev evaluation."""

    round: int
    dev_score: float
    best_dev_score: float
    metric_calls_spent: int
    cost_usd: float
    top_confusions: list[tuple[str, str, int]] = field(default_factory=list)


class ClassificationAdapter:
    """One instance per optimization run."""

    def __init__(
        self,
        spec: TaskSpec,
        task_lm: Any,
        config: Config,
        valset: list[Example],
        trace_path=None,
        on_round: Callable[[RoundInfo], None] | None = None,
        cost_fn: Callable[[], float] | None = None,
    ):
        self.spec = spec
        self.task_lm = task_lm
        self.config = config
        self.label_names = spec.label_names
        self.confusion = ConfusionState()
        self.on_round = on_round
        self.cost_fn = cost_fn or (lambda: 0.0)
        self._val_texts = tuple(e.text for e in valset)
        self._metric_calls = 0
        self._best_dev = 0.0
        self._round = 0

        feats = config.features
        if feats.codebook or feats.confusion_reflection:
            self.proposer: ClassificationProposer | None = ClassificationProposer(
                render_full_prompt=lambda cand: render_prompt(cand, self.label_names),
                confusion_block=(
                    (lambda: render_confusion_block(self.confusion))
                    if feats.confusion_reflection
                    else (lambda: "")
                ),
                trace_path=trace_path,
            )
        else:
            self.proposer = None

    # gepa checks `adapter.propose_new_texts is not None`
    @property
    def propose_new_texts(self):
        return self.proposer

    # ------------------------------------------------------------- evaluate
    def evaluate(
        self, batch: list[Example], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        rendered = render_prompt(candidate, self.label_names)
        preds = classify_batch(
            self.task_lm, rendered, batch, self.label_names, self.config.num_threads
        )
        self._metric_calls += len(batch)
        scores = [1.0 if p.correct else 0.0 for p in preds]

        if self._is_full_val(batch):
            self.confusion.update(preds)
            self._round += 1
            dev_score = sum(scores) / len(scores)
            self._best_dev = max(self._best_dev, dev_score)
            if self.on_round:
                self.on_round(
                    RoundInfo(
                        round=self._round,
                        dev_score=dev_score,
                        best_dev_score=self._best_dev,
                        metric_calls_spent=self._metric_calls,
                        cost_usd=self.cost_fn(),
                        top_confusions=list(self.confusion.last_pairs[:3]),
                    )
                )

        trajectories = None
        if capture_traces:
            trajectories = [
                {"example": ex, "prediction": p} for ex, p in zip(batch, preds, strict=True)
            ]
        return EvaluationBatch(
            outputs=[p.predicted for p in preds], scores=scores, trajectories=trajectories
        )

    def _is_full_val(self, batch: list[Example]) -> bool:
        return len(batch) == len(self._val_texts) and all(
            e.text == t for e, t in zip(batch, self._val_texts, strict=True)
        )

    # ------------------------------------------------- reflective dataset
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        preds: list[Prediction] = [t["prediction"] for t in (eval_batch.trajectories or [])]
        out: dict[str, list[dict[str, Any]]] = {}
        for comp in components_to_update:
            if comp.startswith(LABEL_PREFIX):
                label = comp[len(LABEL_PREFIX):]
                relevant = [p for p in preds if not p.correct and label in (p.gold, p.predicted)]
                # pad with confusion-state examples for this label if the minibatch is clean
                if not relevant:
                    relevant = [
                        p for p in self.confusion.last_preds
                        if not p.correct and label in (p.gold, p.predicted)
                    ][:3]
            else:
                relevant = [p for p in preds if not p.correct]
            if not relevant:  # nothing wrong -> show the minibatch as-is
                relevant = preds
            records = [
                {
                    "Inputs": {"text": p.text},
                    "Generated Outputs": {"response": p.raw},
                    "Feedback": p.feedback,
                }
                for p in relevant[:6]
            ]
            if records:
                out[comp] = records
        if not out:
            raise ValueError("no reflective examples available for any component")
        return out

    # convenience for blob mode default proposer compatibility
    @staticmethod
    def blob_candidate(text: str) -> dict[str, str]:
        return {BLOB_COMPONENT: text}
