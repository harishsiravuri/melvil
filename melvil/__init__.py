"""melvil: optimized, versioned classifier prompts from labeled examples.

Quickstart::

    import melvil as mv

    examples = mv.load_csv("tickets.csv")                  # text,label columns
    train, dev = mv.train_dev_split(examples, dev_size=100, seed=0)
    spec = mv.TaskSpec.from_examples("ticket-intents", examples)
    cfg = mv.Config(task_model="openai/gpt-4.1-mini",
                    reflection_model="openai/gpt-4.1", budget="light")
    artifact = mv.optimize(spec, train, dev, cfg)
    print(artifact.render())                               # deployable prompt
    artifact.save("ticket_intents.v1.json")
"""

from melvil._about import LIBRARY_NAME, __version__
from melvil.adapter import RoundInfo
from melvil.artifact import PromptArtifact, render_prompt
from melvil.config import BUDGET_PRESETS, Config, Features
from melvil.costs import (
    CostEstimate,
    estimate_draft_cost,
    estimate_evaluate_cost,
    estimate_optimize_cost,
)
from melvil.data import Example, load_csv, load_hf, train_dev_split
from melvil.draft import draft
from melvil.evaluate import Report, evaluate, report
from melvil.optimize import optimize
from melvil.screen import ScreenResult, screen
from melvil.taskspec import Label, TaskSpec

__all__ = [
    "BUDGET_PRESETS",
    "LIBRARY_NAME",
    "Config",
    "CostEstimate",
    "Example",
    "Features",
    "Label",
    "PromptArtifact",
    "Report",
    "RoundInfo",
    "TaskSpec",
    "__version__",
    "draft",
    "estimate_draft_cost",
    "estimate_evaluate_cost",
    "estimate_optimize_cost",
    "evaluate",
    "load_csv",
    "load_hf",
    "optimize",
    "render_prompt",
    "report",
    "screen",
    "ScreenResult",
    "train_dev_split",
]
