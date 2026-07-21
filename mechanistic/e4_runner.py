"""E4 — coarse-to-fine arms (C4). Arms: vanilla / full-layer / c2f
(switch=rejections:2; fraction 60/40 as ablation on banking77 only).

Iteration: seeds 30-32, B1, tasks banking77/trec/massive/stance_abortion.
Frozen pass (later, shared campaign confirmation): 5 fresh seeds + B2 cells.
Vanilla B1 reuses E1 cells where they exist (banking77/trec/massive seeds
30-34); stance_abortion vanilla is run here.

Usage: python mechanistic/e4_runner.py --iterate | --arm c2f --task trec --seed 30
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
from prep_data import load_task  # noqa: E402

import melvil as mv  # noqa: E402
from melvil.c2f import optimize_c2f  # noqa: E402

RESULTS_DIR = HERE / "results_e4"
RUNS_DIR = HERE / "runs_e4"
TASKS = ["banking77", "trec", "massive", "stance_abortion"]
ITER_SEEDS = [30, 31, 32]
BUDGET = 800


def run_arm(arm: str, task_name: str, seed: int, budget: int = BUDGET,
            switch=("rejections", 2), tag: str = "iter") -> dict:
    switch_id = f"{switch[0]}{switch[1]}" if arm == "c2f" else ""
    out_path = RESULTS_DIR / f"{tag}_{arm}{switch_id}_{task_name}_b{budget}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    # reuse E1 vanilla cells (identical config) where they exist
    if arm == "vanilla":
        e1 = HERE / "results_e1" / f"gepa_{task_name}_b{budget}_f1_s{seed}.json"
        if e1.exists():
            r = json.loads(e1.read_text())
            rec = {"arm": "vanilla", "task": task_name, "seed": seed, "budget": budget,
                   "test_accuracy": r["test_accuracy"], "cost_usd": 0.0,
                   "source": "reused-e1", "curve": r["curve"]}
            RESULTS_DIR.mkdir(exist_ok=True)
            out_path.write_text(json.dumps(rec, indent=1))
            print(f"[done] {out_path.name}: {rec['test_accuracy']:.3f} (reused E1)")
            return rec

    task = load_task(task_name)
    spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
    fam = FAMILIES["f1"]
    features = {"vanilla": mv.Features.none(), "full": mv.Features.all(),
                "c2f": mv.Features.none()}[arm]
    cfg = mv.Config(task_model=fam["task"], reflection_model=fam["reflection"],
                    budget=budget, seed=seed, features=features,
                    run_dir=str(RUNS_DIR / f"{arm}{switch_id}_{task_name}_b{budget}_s{seed}"))
    if arm == "c2f":
        artifact = optimize_c2f(spec, task["train"], task["dev"], cfg,
                                switch=switch, resume=True)
    else:
        artifact = mv.optimize(spec, task["train"], task["dev"], cfg, resume=True)
    rep = mv.evaluate(artifact, task["test"], model=fam["task"])
    rec = {
        "arm": arm, "switch": list(switch) if arm == "c2f" else None,
        "task": task_name, "seed": seed, "budget": budget,
        "test_accuracy": rep.accuracy, "test_macro_f1": rep.macro_f1,
        "dev_accuracy": artifact.scores.get("dev_accuracy"),
        "notes": artifact.notes, "curve": artifact.curve,
        "cost_usd": round(artifact.budget.get("cost_usd", 0) + rep.cost_usd, 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: test {rep.accuracy:.3f} | {artifact.notes[:60]}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterate", action="store_true")
    ap.add_argument("--arm", choices=("vanilla", "full", "c2f"))
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--fraction-ablation", action="store_true")
    args = ap.parse_args()
    if args.iterate:
        for t in TASKS:
            for s in ITER_SEEDS:
                run_arm("vanilla", t, s)
                run_arm("full", t, s)
                run_arm("c2f", t, s)
        for s in ITER_SEEDS:  # fixed-fraction ablation, banking77 only
            run_arm("c2f", "banking77", s, switch=("fraction", 0.6))
    elif args.arm and args.task:
        sw = ("fraction", 0.6) if args.fraction_ablation else ("rejections", 2)
        run_arm(args.arm, args.task, args.seed, switch=sw)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
