"""MIPROv2 baseline arm (dspy.MIPROv2), for the confirmation table.

Comparability notes (documented in RESULTS):
- MIPROv2 optimizes a dspy program under the ChatAdapter message format
  (structured field markers), not melvil's deployable rendered prompt. We
  evaluate the compiled dspy program in its native format — this measures
  MIPROv2 fairly on its own terms, but the resulting "prompt" is a dspy
  program, not a portable prompt string.
- Budget parity is by preset (auto="light"), not exact metric calls; actual
  task-model calls are measured from lm.history and reported per run.

Usage:  python benchmarks/miprov2_arm.py --task ag_news --seed 0 [--out-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prep_data import TASKS, load_task  # noqa: E402

from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.data import canon  # noqa: E402

TASK_MODEL = "openrouter/openai/gpt-4.1-mini"
PROMPT_MODEL = "openrouter/openai/gpt-4.1"


def run_miprov2(task_name: str, seed: int, out_dir: Path) -> dict:
    out_path = out_dir / f"{task_name}_miprov2_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())

    import dspy

    task = load_task(task_name)
    labels = task["labels"]
    label_list = ", ".join(labels)

    task_lm = dspy.LM(TASK_MODEL, temperature=0.0, max_tokens=40, cache=True)
    prompt_lm = dspy.LM(PROMPT_MODEL, temperature=1.0, max_tokens=8000, cache=True,
                        rollout_id=seed)
    dspy.configure(lm=task_lm)

    sig = dspy.Signature(
        "text -> label",
        f"Classify the given text into one of these categories: {label_list}. "
        "Answer with the category name only.",
    )
    student = dspy.Predict(sig)

    def metric(gold, pred, trace=None):
        return float(canon(getattr(pred, "label", "") or "") == canon(gold.label))

    def to_dspy(rows):
        return [dspy.Example(text=e.text, label=e.label).with_inputs("text") for e in rows]

    trainset, devset, testset = to_dspy(task["train"]), to_dspy(task["dev"]), to_dspy(task["test"])

    task_since, prompt_since = len(task_lm.history), len(prompt_lm.history)
    tele = dspy.MIPROv2(
        metric=metric, auto="light", prompt_model=prompt_lm, task_model=task_lm,
        num_threads=8, seed=seed, verbose=False,
    )
    t0 = time.time()
    compiled = tele.compile(student, trainset=trainset, valset=devset)
    opt_seconds = time.time() - t0

    # single test eval, dspy-native runtime
    test_hist_mark = len(task_lm.history)
    from concurrent.futures import ThreadPoolExecutor

    def run_one(ex):
        try:
            return getattr(compiled(text=ex.text), "label", "") or ""
        except Exception:  # noqa: BLE001
            return ""

    with ThreadPoolExecutor(8) as pool:
        raw_preds = list(pool.map(run_one, testset))
    pairs = [(canon(ex.label), canon(p)) for ex, p in zip(testset, raw_preds, strict=True)]
    test_acc = sum(1 for g, p in pairs if g == p) / len(pairs)
    f1s = []
    for name in {g for g, _ in pairs}:
        tp = sum(1 for g, p in pairs if g == name and p == name)
        fp = sum(1 for g, p in pairs if g != name and p == name)
        fn = sum(1 for g, p in pairs if g == name and p != name)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    test_f1 = sum(f1s) / len(f1s)

    usage_opt = lm_usage(task_lm, task_since)
    usage_prompt = lm_usage(prompt_lm, prompt_since)
    usage_test = lm_usage(task_lm, test_hist_mark)
    result = {
        "task": task_name, "arm": "miprov2", "seed": seed,
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "dev_accuracy": None,  # MIPROv2 tracks its own val internally
        "optimize_task_calls": usage_opt["calls"] - usage_test["calls"],
        "optimize_cost_usd": round(
            usage_cost_usd(usage_opt) + usage_cost_usd(usage_prompt)
            - usage_cost_usd(usage_test), 4),
        "test_cost_usd": round(usage_cost_usd(usage_test), 4),
        "opt_seconds": round(opt_seconds, 1),
        "instructions": compiled.signature.instructions,
        "n_demos": len(getattr(compiled, "demos", []) or []),
        "notes": "dspy-native ChatAdapter runtime; budget parity by preset (auto=light)",
    }
    out_dir.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[done] {out_path.name}: test acc {test_acc:.3f} "
          f"(${result['optimize_cost_usd'] + result['test_cost_usd']:.2f}, "
          f"{result['optimize_task_calls']} opt calls)")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS + ["stance_abortion", "sst5"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results_iter"))
    args = ap.parse_args()
    run_miprov2(args.task, args.seed, Path(args.out_dir))
