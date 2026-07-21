"""Seed-matched GPT-family (F1) vanilla-GEPA baselines at B1, seeds 20-24, all 8
E8 classification tasks. Closes the unpaired gap in the E8 central table: the
frozen F1 GEPA reference reused the v0.1/confirmation vanilla POOL (seeds 0-2,
10-12), which is not seed-matched to the draft arm (seeds 20-24). These runs
give an F1 GEPA arm on the same seeds, enabling a per-seed paired bootstrap.

Saves artifact (for per-example bootstrap) to runs_e8_gepa_f1/{task}/s{seed}/
and a summary to results_e8/gepa_{task}_f1_s{seed}.json.

Run: python mechanistic/gepa_f1_baseline.py [--task banking77]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
sys.path.insert(0, str(HERE))
from e1_frontloading import FAMILIES  # noqa: E402
from prep_data import CONFIRM_EXTRA_TASKS, TASKS, load_task  # noqa: E402

import melvil as mv  # noqa: E402

RESULTS_DIR = HERE / "results_e8"
E8_TASKS = TASKS + CONFIRM_EXTRA_TASKS
SEEDS = [20, 21, 22, 23, 24]


def run(task_name: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"gepa_{task_name}_f1_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    task = load_task(task_name)
    spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
    fam = FAMILIES["f1"]
    cfg = mv.Config(
        task_model=fam["task"], reflection_model=fam["reflection"], budget="light",
        seed=seed, features=mv.Features.none(),
        run_dir=str(HERE / "runs_e8_gepa_f1" / task_name / f"s{seed}"),
    )
    artifact = mv.optimize(spec, task["train"], task["dev"], cfg, resume=True)
    rep = mv.evaluate(artifact, task["test"], model=fam["task"])
    rec = {
        "experiment": "e8_gepa_baseline", "task": task_name, "family": "f1", "seed": seed,
        "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
        "cost_usd": round(artifact.budget.get("cost_usd", 0) + rep.cost_usd, 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: {rep.accuracy:.3f} (${rec['cost_usd']:.2f})")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=E8_TASKS)
    args = ap.parse_args()
    tasks = [args.task] if args.task else E8_TASKS
    total = 0.0
    for t in tasks:
        for s in SEEDS:
            total += run(t, s).get("cost_usd", 0)
    print(f"F1 GEPA baseline spend so far: ${total:.2f}")


if __name__ == "__main__":
    main()
