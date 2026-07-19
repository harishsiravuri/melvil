"""CONFIRMATION PASS — pre-registered protocol, frozen before any results.

Protocol (no changes after seeing numbers; this file is the registration):
- Code: melvil v0.2 (fixed confusion selector, hardened mining). No further
  library changes after this file is committed.
- Datasets (8): the six iteration datasets + two untouched ones
  (tweet_eval/stance_abortion, SetFit/sst5).
- Arms (4): seed prompt (deterministic, 1 eval), vanilla GEPA
  (Features.none()), melvil full (all features), MIPROv2 (auto="light",
  dspy-native runtime — comparability caveat documented in RESULTS).
- Seeds: 10, 11, 12 — FRESH, never used in any prior run.
- Budget: preset "light" for every optimized arm.
- Metrics: TEST accuracy and macro-F1 only (dev is internal to optimization).
  Test touched exactly once per (task, arm, seed).
- Output: results_confirm/ + runs_confirm/; headline table generated from
  these files only.

Usage: python benchmarks/run_confirmation.py --task <name> | --all | --estimate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from miprov2_arm import run_miprov2  # noqa: E402
from prep_data import CONFIRM_EXTRA_TASKS, TASKS, load_task  # noqa: E402
from run_matrix import REFLECTION_MODEL, TASK_MODEL, seed_artifact, spec_for  # noqa: E402

import melvil as mv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CONFIRM_TASKS = TASKS + CONFIRM_EXTRA_TASKS
CONFIRM_SEEDS = [10, 11, 12]
RESULTS_DIR = Path(__file__).parent / "results_confirm"
RUNS_DIR = Path(__file__).parent / "runs_confirm"

ARM_FEATURES = {"vanilla": mv.Features.none(), "full": mv.Features()}


def run_one(task_name: str, arm: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"{task_name}_{arm}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    task = load_task(task_name)
    spec = spec_for(task_name, task)

    if arm == "seed":
        artifact = seed_artifact(spec)
    elif arm == "miprov2":
        return run_miprov2(task_name, seed, RESULTS_DIR)
    else:
        cfg = mv.Config(
            task_model=TASK_MODEL, reflection_model=REFLECTION_MODEL,
            budget="light", seed=seed, features=ARM_FEATURES[arm],
            run_dir=str(RUNS_DIR / task_name / f"{arm}-s{seed}"),
        )
        artifact = mv.optimize(spec, task["train"], task["dev"], cfg, resume=True)

    rep = mv.evaluate(artifact, task["test"], model=TASK_MODEL)  # the single test touch
    result = {
        "task": task_name, "arm": arm, "seed": seed, "codebase": "v0.2-confirm",
        "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
        "unparseable": rep.unparseable,
        "optimize_cost_usd": artifact.budget.get("cost_usd", 0.0),
        "test_cost_usd": rep.cost_usd,
        "artifact_id": artifact.artifact_id,
        "exemplars": len(artifact.exemplars),
        "mining_note": artifact.notes,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[done] {out_path.name}: test {rep.accuracy:.3f} f1 {rep.macro_f1:.3f}")
    return result


def run_task(task_name: str) -> None:
    run_one(task_name, "seed", CONFIRM_SEEDS[0])
    for arm in ["vanilla", "full", "miprov2"]:
        for seed in CONFIRM_SEEDS:
            run_one(task_name, arm, seed)


def estimate() -> None:
    total = 0.0
    for task_name in CONFIRM_TASKS:
        task = load_task(task_name)
        spec = spec_for(task_name, task)
        cfg = mv.Config(task_model=TASK_MODEL, reflection_model=REFLECTION_MODEL,
                        budget="light", features=mv.Features())
        e = mv.estimate_optimize_cost(spec, task["train"], task["dev"], cfg)
        ev = mv.estimate_evaluate_cost(seed_artifact(spec), task["test"], TASK_MODEL)
        n_opt = 3 * len(CONFIRM_SEEDS)  # vanilla, full, miprov2
        n_test = n_opt + 1
        task_total = n_opt * e.total_usd + n_test * ev.total_usd
        total += task_total
        print(f"{task_name}: ~${task_total:.2f}")
    print(f"\nCONFIRMATION ESTIMATE (upper bound): ${total:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=CONFIRM_TASKS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    if args.estimate:
        estimate()
    elif args.task:
        run_task(args.task)
    elif args.all:
        for t in CONFIRM_TASKS:
            run_task(t)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
