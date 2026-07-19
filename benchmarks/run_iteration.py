"""Iteration round (post-v0.1): re-run full vs no_confusion with the FIXED
confusion selector + hardened mining, on the three headroom tasks only.

Question being answered: does confusion-driven reflection earn its keep once
the selector no longer starves label pools?

Runs land in results_iter/ + runs_iter/ (v0.1 matrix results stay untouched).

Usage: python benchmarks/run_iteration.py [--task banking77]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prep_data import load_task  # noqa: E402
from run_matrix import (  # noqa: E402
    ARM_FEATURES,
    SEEDS,
    TASK_MODEL,
    spec_for,
)

import melvil as mv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ITER_TASKS = ["banking77", "ag_news", "massive"]
ITER_ARMS = ["full", "no_confusion"]
RESULTS_DIR = Path(__file__).parent / "results_iter"
RUNS_DIR = Path(__file__).parent / "runs_iter"


def run_one(task_name: str, arm: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"{task_name}_{arm}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    task = load_task(task_name)
    spec = spec_for(task_name, task)
    cfg = mv.Config(
        task_model=TASK_MODEL,
        reflection_model="openrouter/openai/gpt-4.1",
        budget="light",
        seed=seed,
        features=ARM_FEATURES[arm],
        run_dir=str(RUNS_DIR / task_name / f"{arm}-s{seed}"),
    )
    artifact = mv.optimize(spec, task["train"], task["dev"], cfg, resume=True)
    rep = mv.evaluate(artifact, task["test"], model=TASK_MODEL)
    result = {
        "task": task_name, "arm": arm, "seed": seed, "codebase": "v0.2-iter",
        "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
        "dev_accuracy": artifact.scores.get("dev_accuracy"),
        "dev_accuracy_quarantined": artifact.scores.get("dev_accuracy_quarantined"),
        "exemplars": len(artifact.exemplars),
        "mining_note": artifact.notes,
        "optimize_cost_usd": artifact.budget.get("cost_usd", 0.0),
        "test_cost_usd": rep.cost_usd,
        "metric_calls_spent": artifact.budget.get("metric_calls_spent", 0),
        "artifact_id": artifact.artifact_id,
        "curve": artifact.curve,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[done] {out_path.name}: test {rep.accuracy:.3f} "
          f"(${result['optimize_cost_usd'] + result['test_cost_usd']:.2f}) | {artifact.notes}")
    return result


def summarize() -> None:
    import statistics

    print("\n=== iteration summary (v0.2 selector/mining) vs v0.1 matrix ===")
    v01 = Path(__file__).parent / "results"
    for task in ITER_TASKS:
        line = [task]
        for arm in ITER_ARMS:
            new = [json.loads((RESULTS_DIR / f"{task}_{arm}_s{s}.json").read_text())
                   for s in SEEDS if (RESULTS_DIR / f"{task}_{arm}_s{s}.json").exists()]
            old = [json.loads((v01 / f"{task}_{arm}_s{s}.json").read_text())
                   for s in SEEDS if (v01 / f"{task}_{arm}_s{s}.json").exists()]
            if new:
                n = statistics.mean(r["test_accuracy"] for r in new)
                o = statistics.mean(r["test_accuracy"] for r in old) if old else float("nan")
                line.append(f"{arm}: v0.2 {n:.3f} (v0.1 {o:.3f})")
        print(" | ".join(line))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=ITER_TASKS)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    tasks = [args.task] if args.task else ITER_TASKS
    for t in tasks:
        for arm in ITER_ARMS:
            for seed in SEEDS:
                run_one(t, arm, seed)
    summarize()


if __name__ == "__main__":
    main()
