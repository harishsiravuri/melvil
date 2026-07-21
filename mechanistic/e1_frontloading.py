"""E1 — front-loading (C1): do gains concentrate in the first accepted
proposals, across budgets, optimizers, task types, and model families?

Cells: {gepa, miprov2} x {B1=800, B2=1600, B3=3200} x 5 tasks x 5 fresh seeds
(30-34) on F1; B1/B2 cells repeated on F2 (gepa, 3 seeds — cost-trimmed,
reported as such). Tasks: banking77, trec, massive (classification) +
bbh_geometric (BBH geometric_shapes, 250 examples -> 100/50/100 splits,
letter-classification) + gsm8k (QA: numeric answers, custom gepa adapter).
TextGrad: skipped — no maintained harness fit the <1-day budget (stated in
RESULTS).

Instrumentation: melvil `log_proposals` gives every proposal attempt's
metric-call position; artifact curves give accepted candidates. MIPROv2:
trial_logs stored raw (its budget parity is by preset tier — light/medium/
heavy — with actual metric calls measured and reported).

Usage:
  python mechanistic/e1_frontloading.py --prep          # datasets (free)
  python mechanistic/e1_frontloading.py --cell gepa:banking77:800:30:f1
  python mechanistic/e1_frontloading.py --stream <name> # predefined run lists
  python mechanistic/e1_frontloading.py --analyze
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
from prep_data import DATA_DIR, load_task  # noqa: E402

from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.data import Example, stratified_sample  # noqa: E402
from melvil.lmutil import make_lm, text_of  # noqa: E402

RESULTS_DIR = HERE / "results_e1"
RUNS_DIR = HERE / "runs_e1"
CLS_TASKS = ["banking77", "trec", "massive"]
ALL_TASKS = [*CLS_TASKS, "bbh_geometric", "gsm8k"]
BUDGETS = {"B1": 800, "B2": 1600, "B3": 3200}
SEEDS_F1 = [30, 31, 32, 33, 34]
SEEDS_F2 = [30, 31, 32]
FAMILIES = {
    "f1": {"task": "openrouter/openai/gpt-4.1-mini",
           "reflection": "openrouter/openai/gpt-4.1"},
    "f2": {"task": "openrouter/anthropic/claude-haiku-4.5",
           "reflection": "openrouter/anthropic/claude-sonnet-4.5"},
}
GSM8K_SEED_PROMPT = (
    "Solve the math word problem step by step. Show your reasoning, then end your "
    "response with the final numeric answer on its own line in the form '#### <number>'."
)


# ------------------------------------------------------------------ data prep
def prep_bbh() -> None:
    from datasets import load_dataset

    # both BBH hub repos are script-based; load the conversion-branch parquet directly
    url = ("https://huggingface.co/datasets/lukaemon/bbh/resolve/"
           "refs%2Fconvert%2Fparquet/geometric_shapes/test/0000.parquet")
    ds = load_dataset("parquet", data_files={"test": url}, split="test")
    rows = [Example(text=ex["input"], label=ex["target"].strip("() ")) for ex in ds]
    import random
    rng = random.Random(0)
    train_idx = stratified_sample(rows, 100, rng)
    dev_idx = stratified_sample(rows, 50, rng, exclude=set(train_idx))
    used = set(train_idx) | set(dev_idx)
    test_idx = [i for i in range(len(rows)) if i not in used][:100]
    payload = {
        "labels": sorted({r.label for r in rows}),
        "train": [rows[i].__dict__ for i in train_idx],
        "dev": [rows[i].__dict__ for i in dev_idx],
        "test": [rows[i].__dict__ for i in test_idx],
    }
    (DATA_DIR / "bbh_geometric.json").write_text(json.dumps(payload, indent=0))
    print(f"bbh_geometric: {len(payload['labels'])} letters | 100/50/100 "
          "(250-example dataset; deviation from 150/100/300 documented)")


def gsm_answer(ans: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", ans)
    raw = m.group(1) if m else ans.strip().split()[-1]
    return raw.replace(",", "").rstrip(".")


def prep_gsm8k() -> None:
    import random

    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    tr_idx = rng.sample(range(len(train)), 250)
    payload = {
        "labels": [],
        "train": [{"text": train[i]["question"], "label": gsm_answer(train[i]["answer"])}
                  for i in tr_idx[:150]],
        "dev": [{"text": train[i]["question"], "label": gsm_answer(train[i]["answer"])}
                for i in tr_idx[150:250]],
        "test": [{"text": test[i]["question"], "label": gsm_answer(test[i]["answer"])}
                 for i in rng.sample(range(len(test)), 300)],
    }
    (DATA_DIR / "gsm8k.json").write_text(json.dumps(payload, indent=0))
    print("gsm8k: numeric QA | 150/100/300")


# ------------------------------------------------------------ gsm8k QA pieces
def extract_number(response: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", response)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else ""


def solve_batch(lm, prompt: str, examples, num_threads=8):
    from concurrent.futures import ThreadPoolExecutor

    def run(ex):
        try:
            out = lm(messages=[{"role": "system", "content": prompt},
                               {"role": "user", "content": ex.text}])
            raw = text_of(out)
        except Exception as e:  # noqa: BLE001
            raw = f"<error {e}>"
        pred = extract_number(raw)
        ok = pred == ex.label
        fb = ("Correct." if ok else
              f"Incorrect: extracted answer {pred or '(none)'!r}, expected {ex.label!r}. "
              f"Response ended: ...{raw[-200:]}")
        return {"text": ex.text, "gold": ex.label, "pred": pred, "raw": raw,
                "score": 1.0 if ok else 0.0, "feedback": fb}

    with ThreadPoolExecutor(num_threads) as pool:
        return list(pool.map(run, examples))


class QAAdapter:
    """Minimal gepa adapter for GSM8K-style numeric QA (vanilla blob prompt)."""

    propose_new_texts = None  # default gepa proposer

    def __init__(self, task_lm, valset, max_tokens_note=400):
        self.lm = task_lm
        self._val_texts = tuple(e.text for e in valset)
        self.metric_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        preds = solve_batch(self.lm, candidate["instruction"], batch)
        self.metric_calls += len(batch)
        trajs = None
        if capture_traces:
            trajs = [{"example": b, "prediction": p} for b, p in zip(batch, preds, strict=True)]
        return EvaluationBatch(outputs=[p["pred"] for p in preds],
                               scores=[p["score"] for p in preds], trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        out = {}
        for comp in components_to_update:
            recs = []
            for t in (eval_batch.trajectories or []):
                p = t["prediction"]
                if p["score"] < 1.0:
                    recs.append({"Inputs": {"question": p["text"]},
                                 "Generated Outputs": {"response": p["raw"][-400:]},
                                 "Feedback": p["feedback"]})
            out[comp] = (recs or [{"Inputs": {"question": t["prediction"]["text"]},
                                   "Generated Outputs": {"response": t["prediction"]["raw"][-400:]},
                                   "Feedback": t["prediction"]["feedback"]}
                                  for t in (eval_batch.trajectories or [])])[:6]
        return out


def run_gsm8k_gepa(budget: int, seed: int, family: str, run_dir: Path) -> dict:
    import gepa

    task = load_task("gsm8k")
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 400)
    refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    t0, r0 = len(task_lm.history), len(refl_lm.history)
    adapter = QAAdapter(task_lm, task["dev"])
    run_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = run_dir / "proposals.jsonl"
    counter = {"n": 0}

    def refl(prompt: str) -> str:
        with open(proposals_path, "a") as f:
            f.write(json.dumps({"call_idx": counter["n"],
                                "metric_calls_at_call": adapter.metric_calls}) + "\n")
        counter["n"] += 1
        return text_of(refl_lm(prompt))

    result = gepa.optimize(
        seed_candidate={"instruction": GSM8K_SEED_PROMPT},
        trainset=task["train"], valset=task["dev"], adapter=adapter,
        reflection_lm=refl, max_metric_calls=budget, run_dir=str(run_dir),
        seed=seed, display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_prompt = result.candidates[best_idx]["instruction"]
    test_preds = solve_batch(task_lm, best_prompt, task["test"])
    return {
        "curve": [{"metric_calls": int(result.discovery_eval_counts[i]),
                   "dev": round(scores[i], 4)} for i in range(len(scores))],
        "test_accuracy": round(sum(p["score"] for p in test_preds) / len(test_preds), 4),
        "best_prompt": best_prompt,
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(refl_lm, r0)), 4),
        "total_metric_calls": int(result.total_metric_calls or 0),
        "n_proposals": counter["n"],
    }


# ------------------------------------------------------------------ cells
def run_cell(optimizer: str, task_name: str, budget: int, seed: int, family: str) -> dict:
    cell_id = f"{optimizer}_{task_name}_b{budget}_{family}_s{seed}"
    out_path = RESULTS_DIR / f"{cell_id}.json"
    if out_path.exists():
        print(f"[skip] {cell_id}")
        return json.loads(out_path.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)
    run_dir = RUNS_DIR / cell_id

    if task_name == "gsm8k":
        assert optimizer == "gepa", "miprov2+gsm8k handled via dspy path"
        rec = run_gsm8k_gepa(budget, seed, family, run_dir)
    elif optimizer == "gepa":
        import melvil as mv

        task = load_task(task_name)
        spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
        fam = FAMILIES[family]
        cfg = mv.Config(task_model=fam["task"], reflection_model=fam["reflection"],
                        budget=budget, seed=seed, features=mv.Features.none(),
                        run_dir=str(run_dir), log_proposals=True)
        artifact = mv.optimize(spec, task["train"], task["dev"], cfg, resume=True)
        rep = mv.evaluate(artifact, task["test"], model=fam["task"])
        rec = {
            "curve": [{"metric_calls": c["metric_calls"], "dev": c["dev_score"]}
                      for c in artifact.curve],
            "test_accuracy": rep.accuracy,
            "cost_usd": round(artifact.budget.get("cost_usd", 0) + rep.cost_usd, 4),
            "total_metric_calls": artifact.budget.get("metric_calls_spent", 0),
            "n_proposals": (
                len((run_dir / "proposals.jsonl").read_text().splitlines())
                if (run_dir / "proposals.jsonl").exists() else None),
        }
    else:  # miprov2 on classification tasks
        rec = run_miprov2_cell(task_name, budget, seed, family, run_dir)

    rec.update(optimizer=optimizer, task=task_name, budget=budget, seed=seed, family=family)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {cell_id}: test {rec['test_accuracy']:.3f} (${rec['cost_usd']:.2f})")
    return rec


MIPRO_AUTO = {800: "light", 1600: "medium", 3200: "heavy"}


def run_miprov2_cell(task_name, budget, seed, family, run_dir) -> dict:
    import dspy

    from melvil.data import canon

    task = load_task(task_name)
    fam = FAMILIES[family]
    is_qa = task_name == "gsm8k"
    task_lm = dspy.LM(fam["task"], temperature=0.0, max_tokens=400 if is_qa else 40, cache=True)
    prompt_lm = dspy.LM(fam["reflection"], temperature=1.0, max_tokens=8000, cache=True,
                        rollout_id=seed)
    dspy.configure(lm=task_lm)
    if is_qa:
        sig = dspy.Signature("question -> answer", GSM8K_SEED_PROMPT)
        student = dspy.Predict(sig)

        def metric(gold, pred, trace=None):
            return float(extract_number(getattr(pred, "answer", "") or "") == gold.label)

        to_ex = [dspy.Example(question=e.text, label=e.label).with_inputs("question")
                 for e in task["train"]]
        dev_ex = [dspy.Example(question=e.text, label=e.label).with_inputs("question")
                  for e in task["dev"]]
        test_ex = [dspy.Example(question=e.text, label=e.label).with_inputs("question")
                   for e in task["test"]]
    else:
        labels = task["labels"]
        sig = dspy.Signature(
            "text -> label",
            "Classify the given text into one of these categories: "
            + ", ".join(labels) + ". Answer with the category name only.")
        student = dspy.Predict(sig)

        def metric(gold, pred, trace=None):
            return float(canon(getattr(pred, "label", "") or "") == canon(gold.label))

        def dsx(rows):
            return [dspy.Example(text=e.text, label=e.label).with_inputs("text") for e in rows]

        to_ex, dev_ex, test_ex = dsx(task["train"]), dsx(task["dev"]), dsx(task["test"])

    t0, r0 = len(task_lm.history), len(prompt_lm.history)
    tele = dspy.MIPROv2(metric=metric, auto=MIPRO_AUTO[budget], prompt_model=prompt_lm,
                        task_model=task_lm, num_threads=8, seed=seed, verbose=False,
                        track_stats=True)
    compiled = tele.compile(student, trainset=to_ex, valset=dev_ex)

    from concurrent.futures import ThreadPoolExecutor

    def score_one(ex):
        try:
            return metric(ex, compiled(**{k: ex[k] for k in ex.inputs()}))
        except Exception:  # noqa: BLE001
            return 0.0

    with ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(score_one, test_ex))
    trial_logs = getattr(compiled, "trial_logs", None) or getattr(tele, "trial_logs", None)
    return {
        "test_accuracy": round(sum(scores) / len(scores), 4),
        "curve": None,
        "trial_logs": json.loads(json.dumps(trial_logs, default=str)) if trial_logs else None,
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(prompt_lm, r0)), 4),
        "total_metric_calls": lm_usage(task_lm, t0)["calls"],
        "auto_preset": MIPRO_AUTO[budget],
    }


# ------------------------------------------------------------------ streams
STREAMS = {
    # F1 gepa: all tasks x budgets x seeds  (5*3*5 = 75 runs)
    "gepa_f1_a": [("gepa", t, b, s, "f1") for t in ["banking77", "trec"]
                  for b in BUDGETS.values() for s in SEEDS_F1],
    "gepa_f1_b": [("gepa", t, b, s, "f1") for t in ["massive", "bbh_geometric"]
                  for b in BUDGETS.values() for s in SEEDS_F1],
    "gepa_f1_c": [("gepa", "gsm8k", b, s, "f1") for b in BUDGETS.values() for s in SEEDS_F1],
    # F1 miprov2: classification+bbh only (gsm8k-miprov2 dropped: budget parity
    # too loose to interpret; documented)  (4*3*5 = 60 runs)
    "mipro_f1_a": [("miprov2", t, b, s, "f1") for t in ["banking77", "trec"]
                   for b in BUDGETS.values() for s in SEEDS_F1],
    "mipro_f1_b": [("miprov2", t, b, s, "f1") for t in ["massive", "bbh_geometric"]
                   for b in BUDGETS.values() for s in SEEDS_F1],
    # F2 gepa B1/B2, 3 seeds (5*2*3 = 30 runs)
    "gepa_f2": [("gepa", t, b, s, "f2") for t in ALL_TASKS
                for b in (800, 1600) for s in SEEDS_F2],
}


def analyze() -> None:
    rows = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    gep = [r for r in rows if r["optimizer"] == "gepa" and r.get("curve")]
    fracs_by = {}
    for r in gep:
        pts = sorted(r["curve"], key=lambda c: c["metric_calls"])
        seed_dev = pts[0]["dev"]
        best = seed_dev
        gains = []
        for p in pts[1:]:
            new = max(best, p["dev"])
            gains.append(new - best)
            best = new
        total = best - seed_dev
        if total < 0.005 or not gains:
            continue
        key = (r["task"], r["budget"], r["family"])
        fracs_by.setdefault(key, []).append(gains[0] / total)
    print("median fraction of total gain from FIRST accepted proposal:")
    for key in sorted(fracs_by):
        vals = fracs_by[key]
        print(f"  {key}: {statistics.median(vals):.2f} (n={len(vals)})")
    allv = [v for vs in fracs_by.values() for v in vs]
    if allv:
        print(f"  OVERALL: median {statistics.median(allv):.2f}, mean {statistics.mean(allv):.2f} "
              f"(n={len(allv)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--cell")  # optimizer:task:budget:seed:family
    ap.add_argument("--stream", choices=list(STREAMS))
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.prep:
        prep_bbh()
        prep_gsm8k()
    elif args.cell:
        o, t, b, s, f = args.cell.split(":")
        run_cell(o, t, int(b), int(s), f)
    elif args.stream:
        for cell in STREAMS[args.stream]:
            run_cell(*cell)
    elif args.analyze:
        analyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
