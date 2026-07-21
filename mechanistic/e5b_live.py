"""E5 Phase B — LIVE adaptive stopping (C5), the part simulation can't answer:
actual $ saved (reflection spend included) and whether TEST accuracy holds up
better than dev-retention suggests (late accepts may chase dev noise).

Rule (chosen from the enriched Phase-A frontier): stop after the 3rd accepted
candidate OR after 150 metric calls with no new accept, whichever first.
Paired design: banking77/trec/massive × seeds 30–34, F1, vs the E1 vanilla B1
cells (identical seeds/config, full 800 budget).

Usage: python mechanistic/e5b_live.py --all
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
from melvil.adapter import ClassificationAdapter  # noqa: E402
from melvil.artifact import BLOB_COMPONENT, render_prompt  # noqa: E402
from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.lmutil import make_lm, reflection_callable  # noqa: E402
from melvil.optimize import seed_candidate_for  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

RESULTS_DIR = HERE / "results_e5b"
RUNS_DIR = HERE / "runs_e5b"
TASKS = ["banking77", "trec", "massive"]
SEEDS = [30, 31, 32, 33, 34]
BUDGET = 800
STOP_ACCEPTS = 3
PATIENCE = 150


def run_live(task_name: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"{task_name}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    import gepa

    task = load_task(task_name)
    spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
    fam = FAMILIES["f1"]
    task_lm = make_lm(fam["task"], 0.0, 40)
    refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    t0, r0 = len(task_lm.history), len(refl_lm.history)
    cfg = mv.Config(task_model=fam["task"], reflection_model=fam["reflection"],
                    budget=BUDGET, seed=seed, features=mv.Features.none())
    adapter = ClassificationAdapter(spec, task_lm, cfg, valset=task["dev"])
    state = {"last_accept_calls": 100}

    def stopper(gepa_state) -> bool:
        # adapter._round counts full-val evals: round 1 = seed, rounds 2+ = accepts
        accepts = max(0, adapter._round - 1)
        if adapter._round >= 2:
            state.setdefault("round_calls", {})
            if adapter._round not in state["round_calls"]:
                state["round_calls"][adapter._round] = adapter._metric_calls
                state["last_accept_calls"] = adapter._metric_calls
        if accepts >= STOP_ACCEPTS:
            return True
        return adapter._metric_calls - state["last_accept_calls"] > PATIENCE

    run_dir = RUNS_DIR / f"{task_name}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate=seed_candidate_for(spec, mv.Features.none()),
        trainset=task["train"], valset=task["dev"], adapter=adapter,
        reflection_lm=reflection_callable(refl_lm), max_metric_calls=BUDGET,
        stop_callbacks=[stopper], run_dir=str(run_dir), seed=seed,
        display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_prompt = result.candidates[best_idx][BLOB_COMPONENT]
    preds = classify_batch(task_lm, render_prompt({BLOB_COMPONENT: best_prompt},
                                                  spec.label_names),
                           task["test"], spec.label_names, 8)
    test_acc = sum(1 for p in preds if p.correct) / len(preds)
    baseline = json.loads(
        (HERE / "results_e1" / f"gepa_{task_name}_b800_f1_s{seed}.json").read_text())
    rec = {
        "task": task_name, "seed": seed,
        "test_accuracy": round(test_acc, 4),
        "baseline_test_accuracy": baseline["test_accuracy"],
        "metric_calls_spent": adapter._metric_calls,
        "baseline_metric_calls": baseline["total_metric_calls"],
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(refl_lm, r0)), 4),
        "baseline_cost_usd": baseline["cost_usd"],
        "n_candidates": len(scores),
        "best_dev": round(scores[best_idx], 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: test {test_acc:.3f} vs {baseline['test_accuracy']:.3f} | "
          f"{adapter._metric_calls}/{baseline['total_metric_calls']} calls | "
          f"${rec['cost_usd']:.2f} vs ${baseline['cost_usd']:.2f}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--seed", type=int, default=30)
    args = ap.parse_args()
    if args.all:
        for t in TASKS:
            for s in SEEDS:
                run_live(t, s)
    elif args.task:
        run_live(args.task, args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
