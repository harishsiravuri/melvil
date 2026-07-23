"""E9 — does draft() win from the AGGREGATED DIAGNOSIS or from REMOVING THE LOOP?

draft() differs from vanilla GEPA in two ways at once:
  (i)  it sees an aggregated diagnosis (per-label accuracy + top confused pairs
       with concrete examples, computed over the whole dev set in one pass),
       where GEPA sees raw failing examples from a small reflection minibatch;
  (ii) it writes once or twice with NO propose-and-rescore loop, where GEPA
       re-scores after every proposal.

This ablation adds the arm that separates them:

  A. vanilla GEPA                       [reused: results_e8/gepa_{task}_f1_s{20..24}]
  B. GEPA + diagnosis-augmented reflection   [NEW — this script]
  C. draft two-round                    [reused: results_e8/frozen_{task}_f1_s{20..24}]

Arm B is vanilla free-text-blob GEPA (Features.none(); NOT the codebook layer,
NOT per-component targeting) whose reflection prompt is the gepa DEFAULT
template plus the SAME aggregated diagnosis text draft() computes — literally
`melvil.draft.build_diagnosis`, so the information is identical by construction.

BUDGET MATCHING: the diagnosis is built from the predictions GEPA has ALREADY
computed on its most recent full-dev evaluation (the adapter records them in
ConfusionState), so arm B spends zero extra metric calls and is exactly
budget-matched to arm A at B1=800. Everything else is vanilla.

Tasks: banking77 (draft won), trec (draft lost), massive (tie),
stance_abortion (draft's big loss — the global-restructuring task).
F1 family, 5 paired seeds (20-24).

Run: python mechanistic/e9_diag_in_loop.py --task banking77 | --all | --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
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
from melvil.draft import build_diagnosis  # noqa: E402  (the SAME diagnosis draft() uses)
from melvil.lmutil import make_lm, reflection_callable  # noqa: E402
from melvil.optimize import seed_candidate_for  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

RESULTS_DIR = HERE / "results_e9"
RUNS_DIR = HERE / "runs_e9"
E8 = HERE / "results_e8"
ALL_TASKS = ["banking77", "ag_news", "emotion", "trec",
             "clinc150", "massive", "stance_abortion", "sst5"]
# the original suggestive result (kept labeled in RESULTS.md)
ORIGINAL_4 = ["banking77", "trec", "massive", "stance_abortion"]
TASKS = ALL_TASKS
SEEDS = [20, 21, 22, 23, 24]
BUDGET = 800
FAMILIES_RUN = ("f1", "f2")

DIAG_HEADER = (
    "\nAggregated diagnosis of the CURRENT instruction over the full dev set "
    "(per-category accuracy and the most frequent confusions, with concrete "
    "misclassified examples). Use it together with the individual examples above:\n"
)


class DiagnosisAugmentedProposer:
    """gepa's DEFAULT proposal template, verbatim, with the aggregated
    diagnosis block appended. The diagnosis is rebuilt at each reflection from
    the adapter's most recent full-dev predictions — no extra metric calls."""

    def __init__(self, adapter, label_names, refl_call, trace_path: Path):
        self.adapter = adapter
        self.label_names = label_names
        self.refl_call = refl_call
        self.trace_path = trace_path
        self.n_calls = 0
        self.n_with_diagnosis = 0
        self.n_fenced = 0

    def _template(self) -> tuple[str, str]:
        preds = self.adapter.confusion.last_preds
        if not preds:  # before the first full-dev eval (should not happen)
            return InstructionProposalSignature.default_prompt_template, ""
        diagnosis = build_diagnosis(preds, self.label_names)
        block = DIAG_HEADER + diagnosis + "\n"
        tmpl = InstructionProposalSignature.default_prompt_template.replace(
            "Provide the new instructions within ``` blocks.",
            block + "\nProvide the new instructions within ``` blocks.")
        return tmpl, diagnosis

    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            tmpl, diagnosis = self._template()
            prompt = InstructionProposalSignature.prompt_renderer({
                "current_instruction_doc": candidate[name],
                "dataset_with_feedback": reflective_dataset[name],
                "prompt_template": tmpl,
            })
            resp = self.refl_call(prompt)
            new = InstructionProposalSignature.output_extractor(resp)["new_instruction"]
            self.n_calls += 1
            self.n_with_diagnosis += bool(diagnosis)
            self.n_fenced += "```" in resp
            with open(self.trace_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "call_idx": self.n_calls,
                    "diagnosis_present": bool(diagnosis),
                    "diagnosis_chars": len(diagnosis),
                    # parse health: the extractor falls back to the whole
                    # response when no fence is present (affects A and B
                    # equally, but worth monitoring on the Claude family)
                    "fenced": "```" in resp,
                    "reflection_prompt": prompt, "response": resp, "proposed": new,
                }) + "\n")
            out[name] = new
        return out


def run_b(task_name: str, seed: int, family: str = "f1") -> dict:
    out_path = RESULTS_DIR / f"b_{task_name}_{family}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    import gepa

    task = load_task(task_name)
    labels = task["labels"]
    spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 40)
    refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    refl_call = reflection_callable(refl_lm)
    t0, r0 = len(task_lm.history), len(refl_lm.history)

    cfg = mv.Config(task_model=fam["task"], reflection_model=fam["reflection"],
                    budget=BUDGET, seed=seed, features=mv.Features.none())
    adapter = ClassificationAdapter(spec, task_lm, cfg, valset=task["dev"])
    run_dir = RUNS_DIR / (f"{task_name}_s{seed}" if family == "f1" and task_name in ORIGINAL_4
                          else f"{task_name}_{family}_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    proposer = DiagnosisAugmentedProposer(adapter, labels, refl_call,
                                          run_dir / "traces.jsonl")
    adapter.proposer = proposer  # exposed to gepa via .propose_new_texts

    result = gepa.optimize(
        seed_candidate=seed_candidate_for(spec, mv.Features.none()),
        trainset=task["train"], valset=task["dev"], adapter=adapter,
        reflection_lm=refl_call, max_metric_calls=BUDGET,
        run_dir=str(run_dir), seed=seed, display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_prompt = result.candidates[best_idx][BLOB_COMPONENT]
    rendered = render_prompt({BLOB_COMPONENT: best_prompt}, labels)
    test_preds = classify_batch(task_lm, rendered, task["test"], labels, 8)
    test_acc = sum(1 for p in test_preds if p.correct) / len(test_preds)

    rec = {
        "experiment": "e9", "arm": "b_diag_in_loop", "task": task_name,
        "family": family, "seed": seed,
        "test_accuracy": round(test_acc, 4),
        "test_correct": [1 if p.correct else 0 for p in test_preds],
        "seed_dev": round(scores[0], 4), "best_dev": round(scores[best_idx], 4),
        "n_candidates": len(scores),
        "reflection_calls": proposer.n_calls,
        "reflections_with_diagnosis": proposer.n_with_diagnosis,
        "reflections_fenced": proposer.n_fenced,
        "metric_calls_spent": adapter._metric_calls,
        "best_prompt": best_prompt,
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(refl_lm, r0)), 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: test {test_acc:.3f} | {proposer.n_with_diagnosis}"
          f"/{proposer.n_calls} reflections carried the diagnosis | "
          f"{adapter._metric_calls} calls | ${rec['cost_usd']:.2f}")
    return rec


def _ci95(v):
    return 1.96 * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def _load(task: str, family: str) -> dict:
    """{seed: {A,B,C}} for one (task, family)."""
    out = {}
    for sd in SEEDS:
        a = E8 / f"gepa_{task}_{family}_s{sd}.json"
        c = E8 / f"frozen_{task}_{family}_s{sd}.json"
        b = RESULTS_DIR / f"b_{task}_{family}_s{sd}.json"
        if a.exists() and c.exists() and b.exists():
            out[sd] = {
                "A": json.loads(a.read_text())["test_accuracy"],
                "B": json.loads(b.read_text())["test_accuracy"],
                "C": json.loads(c.read_text())["arms"]["x2"]["test_accuracy"],
            }
    return out


def _stat(diffs: list[float]) -> dict:
    m = statistics.mean(diffs)
    ci = _ci95(diffs)
    # Degeneracy guard: on a saturated task every seed can give the IDENTICAL
    # paired difference, collapsing the CI to zero width. That is an artifact
    # of zero variance, not evidence of an effect, so it must not be scored as
    # a win or a loss. (Rule added while runs were still in flight, on the
    # principle -- not tuned to an outcome.)
    degenerate = len(diffs) > 1 and statistics.pstdev(diffs) == 0.0
    if degenerate:
        verdict = "T"
    else:
        verdict = "W" if m - ci > 0 else ("L" if m + ci < 0 else "T")
    return {"mean": round(m, 4), "ci95": round(ci, 4),
            "lo": round(m - ci, 4), "hi": round(m + ci, 4),
            "verdict": verdict, "degenerate": degenerate}


def analyze() -> None:
    out = {"seeds": SEEDS, "budget": BUDGET, "per_family": {}}
    for fam in FAMILIES_RUN:
        rows, ba_all, bc_all, ba_4, wtl = {}, [], [], [], {"W": [], "T": [], "L": []}
        for t in ALL_TASKS:
            d = _load(t, fam)
            if not d:
                continue
            seeds = sorted(d)
            A = [d[s2]["A"] for s2 in seeds]
            B = [d[s2]["B"] for s2 in seeds]
            C = [d[s2]["C"] for s2 in seeds]
            ba = [d[s2]["B"] - d[s2]["A"] for s2 in seeds]
            bc = [d[s2]["B"] - d[s2]["C"] for s2 in seeds]
            ba_all += ba
            bc_all += bc
            if t in ORIGINAL_4 and fam == "f1":
                ba_4 += ba
            rows[t] = {"n_seeds": len(seeds),
                       "A": round(statistics.mean(A), 4), "B": round(statistics.mean(B), 4),
                       "C": round(statistics.mean(C), 4),
                       "B_minus_A": _stat(ba), "B_minus_C": _stat(bc)}
            wtl[rows[t]["B_minus_A"]["verdict"]].append(t)
        if not rows:
            continue
        out["per_family"][fam] = {
            "per_task": rows,
            "pooled_B_minus_A": _stat(ba_all) if ba_all else None,
            "pooled_B_minus_C": _stat(bc_all) if bc_all else None,
            "n_pooled": len(ba_all),
            "wtl_B_vs_A": {k: v for k, v in wtl.items()},
            "n_tasks": len(rows),
        }
        if ba_4:
            out["per_family"][fam]["original_4task_B_minus_A"] = _stat(ba_4)

    for fam, fd in out["per_family"].items():
        print(f"\n===== family {fam} ({fd['n_tasks']} tasks) =====")
        print(f"{'task':<17}{'A':>8}{'B':>8}{'C':>8}   {'B-A [95% CI]':>24} {'B-C [95% CI]':>24}")
        for t, r in fd["per_task"].items():
            ba, bc = r["B_minus_A"], r["B_minus_C"]
            print(f"{t:<17}{r['A']:>8.3f}{r['B']:>8.3f}{r['C']:>8.3f}   "
                  f"{ba['mean']:>+8.3f} [{ba['lo']:+.3f},{ba['hi']:+.3f}]{ba['verdict']:>2} "
                  f"{bc['mean']:>+8.3f} [{bc['lo']:+.3f},{bc['hi']:+.3f}]{bc['verdict']:>2}")
        p1, p2 = fd["pooled_B_minus_A"], fd["pooled_B_minus_C"]
        print(f"POOLED (n={fd['n_pooled']}): B-A {p1['mean']:+.4f} ± {p1['ci95']:.4f} "
              f"[{p1['verdict']}] | B-C {p2['mean']:+.4f} ± {p2['ci95']:.4f} [{p2['verdict']}]")
        w = fd["wtl_B_vs_A"]
        print(f"W/T/L (B vs A): {len(w['W'])}/{len(w['T'])}/{len(w['L'])}"
              f"  W={w['W']} L={w['L']}")
        if fd.get("original_4task_B_minus_A"):
            o = fd["original_4task_B_minus_A"]
            print(f"  [original 4-task B-A for reference: {o['mean']:+.4f} ± {o['ci95']:.4f}]")

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "e9_full.json").write_text(json.dumps(out, indent=1))
    spend = sum(json.loads(q.read_text()).get("cost_usd", 0)
                for q in RESULTS_DIR.glob("b_*.json"))
    print(f"\ntotal arm-B spend: ${spend:.2f}\nwrote results_sim/e9_full.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=ALL_TASKS)
    ap.add_argument("--family", default="f1", choices=FAMILIES_RUN)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.task:
        for sd in SEEDS:
            run_b(args.task, sd, args.family)
    elif args.all:
        for t in ALL_TASKS:
            for sd in SEEDS:
                run_b(t, sd, args.family)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
