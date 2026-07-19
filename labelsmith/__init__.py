"""labelsmith: optimized, versioned classifier prompts from labeled examples.

Quickstart::

    import labelsmith as ls

    examples = ls.load_csv("tickets.csv")                  # text,label columns
    train, dev = ls.train_dev_split(examples, dev_size=100, seed=0)
    spec = ls.TaskSpec.from_examples("ticket-intents", examples)
    cfg = ls.Config(task_model="openai/gpt-4.1-mini",
                    reflection_model="openai/gpt-4.1", budget="light")
    artifact = ls.optimize(spec, train, dev, cfg)
    print(artifact.render())                               # deployable prompt
    artifact.save("ticket_intents.v1.json")
"""

from labelsmith._about import LIBRARY_NAME, __version__
from labelsmith.adapter import RoundInfo
from labelsmith.artifact import PromptArtifact, render_prompt
from labelsmith.config import BUDGET_PRESETS, Config, Features
from labelsmith.costs import CostEstimate, estimate_evaluate_cost, estimate_optimize_cost
from labelsmith.data import Example, load_csv, load_hf, train_dev_split
from labelsmith.evaluate import Report, evaluate, report
from labelsmith.optimize import optimize
from labelsmith.taskspec import Label, TaskSpec

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
    "estimate_evaluate_cost",
    "estimate_optimize_cost",
    "evaluate",
    "load_csv",
    "load_hf",
    "optimize",
    "render_prompt",
    "report",
    "train_dev_split",
]
