"""The benchmark matrix: datasets x arms x seeds, driven by the library itself.

Arms (FIXED before any results, per the benchmark discipline):
  seed          the unoptimized seed prompt, evaluated once (deterministic)
  vanilla       Features.none() — plain GEPA, single instruction blob
  full          the complete classification layer
  no_codebook   full minus per-label codebook (blob prompt, confusion block on)
  no_confusion  full minus confusion-driven reflection
  no_mining     full minus hard-example mining
Optional arm (run only if budget allows, decided at the gate): miprov2 — not
implemented in v0.1; recorded as not-run if absent.

3 seeds per optimized arm; budget preset "light" everywhere; dev drives all
optimization; each task's test split is touched exactly once per (arm, seed),
at the end. Metrics: test accuracy, macro-F1, cost. Transfer: the `full` arm's
artifacts re-evaluated on a cheaper model.

Usage:
  python benchmarks/run_matrix.py --estimate            # cost gate printout
  python benchmarks/run_matrix.py --task ag_news        # one task, all arms/seeds
  python benchmarks/run_matrix.py --all                 # everything
  python benchmarks/run_matrix.py --transfer            # transfer evals (after --all)
Every run checkpoints to benchmarks/results/ and is skipped if present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prep_data import TASKS, load_task  # noqa: E402

import melvil as mv  # noqa: E402
from melvil.artifact import PromptArtifact  # noqa: E402
from melvil.optimize import seed_candidate_for  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = Path(__file__).parent / "runs"

TASK_MODEL = "openrouter/openai/gpt-4.1-mini"
REFLECTION_MODEL = "openrouter/openai/gpt-4.1"
TRANSFER_MODEL = "openrouter/openai/gpt-4.1-nano"
BUDGET = "light"
SEEDS = [0, 1, 2]

ARM_FEATURES: dict[str, mv.Features] = {
    "vanilla": mv.Features.none(),
    "full": mv.Features.all(),
    "no_codebook": mv.Features(
        codebook=False, confusion_reflection=True, hard_example_mining=True
    ),
    "no_confusion": mv.Features(
        codebook=True, confusion_reflection=False, hard_example_mining=True
    ),
    "no_mining": mv.Features(
        codebook=True, confusion_reflection=True, hard_example_mining=False
    ),
}
ARMS = ["seed", *ARM_FEATURES]


def spec_for(task_name: str, task: dict) -> mv.TaskSpec:
    return mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])


def config_for(arm: str, seed: int, task_name: str) -> mv.Config:
    return mv.Config(
        task_model=TASK_MODEL,
        reflection_model=REFLECTION_MODEL,
        budget=BUDGET,
        seed=seed,
        features=ARM_FEATURES[arm],
        run_dir=str(RUNS_DIR / task_name / f"{arm}-s{seed}"),
    )


def seed_artifact(spec: mv.TaskSpec) -> PromptArtifact:
    """The unoptimized baseline: the blob-mode seed prompt as an artifact."""
    components = seed_candidate_for(spec, mv.Features.none())
    return PromptArtifact(
        task_name=spec.name, components=components, label_names=spec.label_names,
        models={"task": TASK_MODEL}, notes="unoptimized seed prompt (arm: seed)",
    )


def run_one(task_name: str, arm: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"{task_name}_{arm}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    task = load_task(task_name)
    spec = spec_for(task_name, task)

    if arm == "seed":
        artifact = seed_artifact(spec)
        dev_acc = None
    else:
        artifact = mv.optimize(
            spec, task["train"], task["dev"], config_for(arm, seed, task_name), resume=True
        )
        dev_acc = artifact.scores.get("dev_accuracy")

    rep = mv.evaluate(artifact, task["test"], model=TASK_MODEL)  # the single test touch
    result = {
        "task": task_name, "arm": arm, "seed": seed,
        "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
        "unparseable": rep.unparseable,
        "dev_accuracy": dev_acc,
        "optimize_cost_usd": artifact.budget.get("cost_usd", 0.0),
        "test_cost_usd": rep.cost_usd,
        "metric_calls_spent": artifact.budget.get("metric_calls_spent", 0),
        "artifact_id": artifact.artifact_id,
        "exemplars": len(artifact.exemplars),
        "curve": artifact.curve,
        "top_confusions": [list(t) for t in rep.top_confusions[:5]],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[done] {out_path.name}: test acc {rep.accuracy:.3f} f1 {rep.macro_f1:.3f} "
          f"(${result['optimize_cost_usd'] + result['test_cost_usd']:.2f})")
    return result


def run_task(task_name: str) -> None:
    run_one(task_name, "seed", 0)
    for arm in ARM_FEATURES:
        for seed in SEEDS:
            run_one(task_name, arm, seed)


def run_transfer(task_name: str) -> None:
    """Evaluate the `full` arm's artifacts on the cheaper transfer model."""
    task = load_task(task_name)
    for seed in SEEDS:
        out_path = RESULTS_DIR / f"{task_name}_transfer_s{seed}.json"
        if out_path.exists():
            print(f"[skip] {out_path.name}")
            continue
        art_path = RUNS_DIR / task_name / f"full-s{seed}" / "artifact.json"
        if not art_path.exists():
            print(f"[missing artifact] {art_path} — run the full arm first")
            continue
        artifact = PromptArtifact.load(art_path)
        rep = mv.evaluate(artifact, task["test"], model=TRANSFER_MODEL)
        out_path.write_text(json.dumps({
            "task": task_name, "arm": "transfer", "seed": seed,
            "transfer_model": TRANSFER_MODEL,
            "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
            "test_cost_usd": rep.cost_usd, "artifact_id": artifact.artifact_id,
        }, indent=1))
        print(f"[done] {out_path.name}: {rep.accuracy:.3f} on {TRANSFER_MODEL}")


def estimate() -> None:
    total = 0.0
    for task_name in TASKS:
        task = load_task(task_name)
        spec = spec_for(task_name, task)
        cfg = config_for("full", 0, task_name)
        e = mv.estimate_optimize_cost(spec, task["train"], task["dev"], cfg)
        ev = mv.estimate_evaluate_cost(seed_artifact(spec), task["test"], TASK_MODEL)
        n_opt = len(ARM_FEATURES) * len(SEEDS)
        n_test = n_opt + 1 + len(SEEDS)  # + seed arm + transfer evals
        task_total = n_opt * e.total_usd + n_test * ev.total_usd
        total += task_total
        print(f"{task_name}: {n_opt} optimize runs x ${e.total_usd:.2f} "
              f"+ {n_test} test evals x ${ev.total_usd:.2f} = ${task_total:.2f}")
    print(f"\nFULL MATRIX ESTIMATE (upper bound, ignores caching): ${total:.2f}")
    print("Gate: ask before launching if > $40 (and any launch > $10 needs an explicit OK).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    args = ap.parse_args()
    if args.estimate:
        estimate()
    elif args.transfer:
        for t in TASKS:
            run_transfer(t)
    elif args.task:
        run_task(args.task)
    elif args.all:
        for t in TASKS:
            run_task(t)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
