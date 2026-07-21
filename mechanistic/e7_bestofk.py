"""E7 — best-of-K first rewrite (C7): invest the budget where the (folk) story
says the value is. Round 1 only: sample K=4 diverse whole-prompt rewrites
(different reflection cache namespaces AND different error examples shown),
score each on a fixed 25-example dev screen, promote the best, then continue
vanilla GEPA at matched TOTAL budget (screen + error-gathering calls deducted).

Context from E1: the first accepted proposal carries only a median ~33% of
GEPA's gain, so expectations here are tempered — E7 now tests whether
better round-1 investment helps AT ALL, closing the loop with data.

Usage: python mechanistic/e7_bestofk.py --task banking77 --seed 30
       python mechanistic/e7_bestofk.py --all   (banking77/trec/massive x seeds 30-34, F1)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
sys.path.insert(0, str(HERE))
from e1_frontloading import FAMILIES  # noqa: E402
from gepa.strategies.instruction_proposal import InstructionProposalSignature  # noqa: E402
from prep_data import load_task  # noqa: E402

import melvil as mv  # noqa: E402
from melvil.adapter import ClassificationAdapter  # noqa: E402
from melvil.artifact import BLOB_COMPONENT, render_prompt  # noqa: E402
from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.lmutil import make_lm, reflection_callable  # noqa: E402
from melvil.optimize import seed_candidate_for  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

RESULTS_DIR = HERE / "results_e7"
RUNS_DIR = HERE / "runs_e7"
K = 4
SCREEN_N = 25
ERR_POOL_N = 24  # train examples evaluated to harvest error examples
BUDGET = 800
TASKS = ["banking77", "trec", "massive"]
SEEDS = [30, 31, 32, 33, 34]


def run_e7(task_name: str, seed: int, family: str = "f1") -> dict:
    out_path = RESULTS_DIR / f"{task_name}_{family}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    import gepa

    task = load_task(task_name)
    spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 40)
    t0 = len(task_lm.history)
    refl_cost_usd = 0.0
    spent = 0

    seed_cand = seed_candidate_for(spec, mv.Features.none())
    seed_prompt_rendered = render_prompt(seed_cand, spec.label_names)

    # ------- harvest error examples from a small train sample
    rng = random.Random(seed)
    pool = rng.sample(task["train"], ERR_POOL_N)
    pool_preds = classify_batch(task_lm, seed_prompt_rendered, pool, spec.label_names, 8)
    spent += ERR_POOL_N
    errors = [p for p in pool_preds if not p.correct]
    if len(errors) < 3:
        errors = pool_preds  # degenerate: few errors, show a mixed sample

    # ------- K diverse rewrites
    screen = task["dev"][:SCREEN_N]
    candidates = []
    for k in range(K):
        refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed * 100 + k)
        r0 = len(refl_lm.history)
        sample = rng.sample(errors, min(3, len(errors)))
        reflective = [{"Inputs": {"text": p.text},
                       "Generated Outputs": {"response": p.raw},
                       "Feedback": p.feedback} for p in sample]
        prompt = InstructionProposalSignature.prompt_renderer({
            "current_instruction_doc": seed_cand[BLOB_COMPONENT],
            "dataset_with_feedback": reflective,
        })
        resp = reflection_callable(refl_lm)(prompt)
        new_instruction = InstructionProposalSignature.output_extractor(resp)["new_instruction"]
        preds = classify_batch(
            task_lm, render_prompt({BLOB_COMPONENT: new_instruction}, spec.label_names),
            screen, spec.label_names, 8)
        spent += SCREEN_N
        acc = sum(1 for p in preds if p.correct) / len(preds)
        refl_cost_usd += usage_cost_usd(lm_usage(refl_lm, r0))
        candidates.append({"k": k, "screen_acc": acc, "instruction": new_instruction})

    winner = max(candidates, key=lambda c: c["screen_acc"])

    # ------- continue vanilla GEPA from the winner at the remaining budget
    remaining = BUDGET - spent
    cfg = mv.Config(task_model=fam["task"], reflection_model=fam["reflection"],
                    budget=remaining, seed=seed, features=mv.Features.none())
    adapter = ClassificationAdapter(spec, task_lm, cfg, valset=task["dev"])
    refl_lm_main = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    rm0 = len(refl_lm_main.history)
    run_dir = RUNS_DIR / f"{task_name}_{family}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate={BLOB_COMPONENT: winner["instruction"]},
        trainset=task["train"], valset=task["dev"], adapter=adapter,
        reflection_lm=reflection_callable(refl_lm_main), max_metric_calls=remaining,
        run_dir=str(run_dir), seed=seed, display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_prompt = result.candidates[best_idx][BLOB_COMPONENT]
    test_preds = classify_batch(
        task_lm, render_prompt({BLOB_COMPONENT: best_prompt}, spec.label_names),
        task["test"], spec.label_names, 8)
    test_acc = sum(1 for p in test_preds if p.correct) / len(test_preds)

    rec = {
        "experiment": "e7", "task": task_name, "family": family, "seed": seed,
        "screen_accs": [c["screen_acc"] for c in candidates],
        "winner_k": winner["k"], "winner_screen_acc": winner["screen_acc"],
        "round1_dev_first_full": round(scores[0], 4),  # winner's full-dev score
        "best_dev": round(scores[best_idx], 4),
        "test_accuracy": round(test_acc, 4),
        "budget_screen_spent": spent, "budget_gepa": remaining,
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0)) + refl_cost_usd
                          + usage_cost_usd(lm_usage(refl_lm_main, rm0)), 4),
        "curve": [{"metric_calls": spent + int(result.discovery_eval_counts[i]),
                   "dev": round(scores[i], 4)} for i in range(len(scores))],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: test {test_acc:.3f} "
          f"(winner k={winner['k']} screen {winner['screen_acc']:.2f}; ${rec['cost_usd']:.2f})")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for t in TASKS:
            for s in SEEDS:
                run_e7(t, s)
    elif args.task:
        run_e7(args.task, args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
