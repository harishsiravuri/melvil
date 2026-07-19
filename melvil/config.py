"""Run configuration: models, budget presets, feature toggles, seeds.

Plain dataclasses — no YAML required, but `Config.from_yaml()` exists for teams
that want files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Budget presets in metric calls (one metric call = one task-model
#: classification). "light" matches the dspy auto="light" budget for a
#: single-predictor program with a 100-example dev set (measured in the pilot).
BUDGET_PRESETS: dict[str, int] = {"light": 800, "medium": 2000, "heavy": 5000}


@dataclass
class Features:
    """The classification layer. Each feature is independently toggleable
    (benchmark ablations rely on this).

    - codebook: structure the prompt as named components (task instruction,
      one definition per label, boundary rules) and let GEPA evolve them
      per-component. Off = single free-text instruction blob (vanilla GEPA).
    - confusion_reflection: compute the dev confusion matrix as the run
      progresses, target the most-confused labels' components for update, and
      show the reflection LM the top confused pairs with concrete misclassified
      examples.
    - hard_example_mining: persistently misclassified dev examples become
      candidate few-shot exemplars, selected to cover the top confused
      boundaries; kept only if they don't hurt dev accuracy (one extra dev
      eval, reserved out of the budget).
    """

    codebook: bool = True
    confusion_reflection: bool = True
    hard_example_mining: bool = True

    @classmethod
    def none(cls) -> Features:
        """Vanilla GEPA: all classification features off."""
        return cls(codebook=False, confusion_reflection=False, hard_example_mining=False)


@dataclass
class Config:
    """Everything `optimize()` needs. `budget` is a preset name or an explicit
    metric-call cap."""

    task_model: str
    reflection_model: str
    budget: int | str = "light"
    seed: int = 0
    features: Features = field(default_factory=Features)
    num_threads: int = 8
    run_dir: str | None = None
    task_temperature: float = 0.0
    task_max_tokens: int = 40
    reflection_temperature: float = 1.0
    reflection_max_tokens: int = 8000
    reflection_minibatch_size: int = 3
    max_exemplars: int = 6
    mining_llm_screen: bool = True
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # $/M in, $/M out

    @property
    def metric_calls(self) -> int:
        if isinstance(self.budget, int):
            return self.budget
        try:
            return BUDGET_PRESETS[self.budget]
        except KeyError:
            raise ValueError(
                f"budget must be an int or one of {sorted(BUDGET_PRESETS)}; got {self.budget!r}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metric_calls"] = self.metric_calls
        return d

    @property
    def config_hash(self) -> str:
        """Stable hash of everything that affects the optimization result
        (run_dir and thread count excluded)."""
        d = self.to_dict()
        d.pop("run_dir", None)
        d.pop("num_threads", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        if "features" in d and isinstance(d["features"], dict):
            d["features"] = Features(**d["features"])
        if "prices" in d:
            d["prices"] = {k: tuple(v) for k, v in d["prices"].items()}
        return cls(**d)
