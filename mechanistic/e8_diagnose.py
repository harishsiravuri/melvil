"""E8 — diagnose-then-write (C8): is evolution even necessary?

One structured error diagnosis of the seed prompt on dev + ONE reflection-model
call that writes the full prompt directly. Optional second iteration (x2).
Positioning: Prompt-MII (arXiv:2510.16932) induces instructions one-shot with an
RL-trained model; this is training-free and error-grounded.

Cost per run ~2 dev evals + 1-2 reflection calls + 1 test eval. Test is touched
once per (task, arm, seed), at the end.

Usage:
  python mechanistic/e8_diagnose.py --task banking77 --seed 0 [--family f1]
  python mechanistic/e8_diagnose.py --iterate     # cheap 2-task x 2-seed pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
from prep_data import CONFIRM_EXTRA_TASKS, TASKS, load_task  # noqa: E402

from melvil.artifact import OUTPUT_CONTRACT  # noqa: E402
from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.data import Example  # noqa: E402
from melvil.evaluate import accuracy, macro_f1, per_label_stats  # noqa: E402
from melvil.features.confusion import top_confused_pairs  # noqa: E402
from melvil.lmutil import make_lm, text_of  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

FAMILIES = {
    "f1": {"task": "openrouter/openai/gpt-4.1-mini",
           "reflection": "openrouter/openai/gpt-4.1"},
    # f2 chosen at freeze time (gate discussion): a second model family
    "f2": {"task": "openrouter/anthropic/claude-haiku-4.5",
           "reflection": "openrouter/anthropic/claude-sonnet-4.5"},
}
ALL_TASKS = TASKS + CONFIRM_EXTRA_TASKS
RESULTS_DIR = HERE / "results_e8"

REWRITE_TEMPLATE = """You are an expert prompt engineer. A text classifier runs with the SYSTEM PROMPT below. It was evaluated on {n} held-out examples; a structured error diagnosis follows.

CURRENT SYSTEM PROMPT:
```
{prompt}
```

DIAGNOSIS (accuracy {acc:.1%}):
{diagnosis}

Write a NEW, complete replacement system prompt that fixes the diagnosed failure modes. Requirements:
- It must list every category name exactly as given: {labels}.
- Add whatever category definitions, decision rules, and tie-breakers the errors call for; keep it compact enough to follow reliably.
- Preserve this exact final line, verbatim, as the last line: "{contract}"
- Do not invent categories, do not include few-shot examples.

Return ONLY the new system prompt inside one fenced code block."""


def build_diagnosis(preds, label_names) -> str:
    stats = per_label_stats(preds, label_names)
    lines = ["Per-category recall (worst first):"]
    for name, s in sorted(stats.items(), key=lambda kv: kv[1]["recall"]):
        if s["support"]:
            lines.append(f"- {name}: recall {s['recall']:.2f}, precision {s['precision']:.2f} "
                         f"(n={s['support']})")
    pairs = top_confused_pairs(preds, k=4)
    if pairs:
        lines.append("\nMost frequent confusions (true -> predicted, with examples):")
        for gold, pred_lab, count in pairs:
            shown = [p for p in preds if not p.correct and p.gold == gold
                     and p.predicted == pred_lab][:3]
            lines.append(f"- '{gold}' misread as '{pred_lab or '(unparseable)'}' x{count}:")
            for p in shown:
                lines.append(f"    TEXT: {p.text[:200]}")
    n_unp = sum(1 for p in preds if not p.predicted)
    if n_unp:
        lines.append(f"\n{n_unp} responses failed to parse as a category name.")
    return "\n".join(lines)


def extract_prompt(response: str, label_names) -> str:
    import re

    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    prompt = max(blocks, key=len).strip() if blocks else response.strip()
    if OUTPUT_CONTRACT not in prompt:
        prompt = prompt.rstrip() + "\n\n" + OUTPUT_CONTRACT
    return prompt


def seed_prompt_for(task: dict) -> str:
    return ("Classify the given text into one of these categories: "
            + ", ".join(task["labels"]) + ".\n\n" + OUTPUT_CONTRACT)


def run_e8(task_name: str, seed: int, family: str, iterations: int = 2,
           tag: str = "iter") -> dict:
    out_path = RESULTS_DIR / f"{tag}_{task_name}_{family}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    task = load_task(task_name)
    labels = task["labels"]
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 40)
    refl_lm = make_lm(fam["reflection"], 1.0, 4000, rollout_id=seed)
    t_since, r_since = len(task_lm.history), len(refl_lm.history)

    prompts = {"seed": seed_prompt_for(task)}
    dev_accs = {}
    current = prompts["seed"]
    for it in range(1, iterations + 1):
        dev_preds = classify_batch(task_lm, current, task["dev"], labels, 8)
        dev_accs[f"input_of_x{it}"] = accuracy(dev_preds)
        diagnosis = build_diagnosis(dev_preds, labels)
        rewrite = text_of(refl_lm(REWRITE_TEMPLATE.format(
            n=len(task["dev"]), prompt=current, acc=accuracy(dev_preds),
            diagnosis=diagnosis, labels=", ".join(labels), contract=OUTPUT_CONTRACT,
        )))
        current = extract_prompt(rewrite, labels)
        prompts[f"x{it}"] = current

    # single test eval per arm, at the end
    results = {}
    for arm, prompt in prompts.items():
        preds = classify_batch(task_lm, prompt, task["test"], labels, 8)
        stats = per_label_stats(preds, labels)
        results[arm] = {"test_accuracy": accuracy(preds), "test_macro_f1": macro_f1(stats),
                        "unparseable": sum(1 for p in preds if not p.predicted)}

    tu, ru = lm_usage(task_lm, t_since), lm_usage(refl_lm, r_since)
    record = {
        "experiment": "e8", "phase": tag, "task": task_name, "family": family, "seed": seed,
        "arms": results, "dev_accs": dev_accs,
        "prompts": prompts,
        "cost_usd": round(usage_cost_usd(tu) + usage_cost_usd(ru), 4),
        "task_lm_calls": tu["calls"], "reflection_calls": ru["calls"],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1))
    arms_s = " | ".join(f"{a}: {r['test_accuracy']:.3f}" for a, r in results.items())
    print(f"[done] {out_path.name}: {arms_s} (${record['cost_usd']:.2f})")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=ALL_TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="f1", choices=list(FAMILIES))
    ap.add_argument("--iterate", action="store_true",
                    help="cheap iteration pass: banking77+trec, seeds 0/1, f1")
    args = ap.parse_args()
    if args.iterate:
        for t in ("banking77", "trec"):
            for s in (0, 1):
                run_e8(t, s, "f1", tag="iter")
    elif args.task:
        run_e8(args.task, args.seed, args.family)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
