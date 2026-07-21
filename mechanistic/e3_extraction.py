"""E3 (extraction family) — per-field structure vs free-text prompt (C3).

Task: MIT Restaurant slot filling (tner/mit_restaurant) rendered text-in /
JSON-out. Metric: micro slot F1 (type-sensitive multiset span matching);
per-example F1 drives GEPA. Arms: free-text blob vs per-field codebook
(task_instruction + field::<slot> definition/format + boundary_rules),
GEPA B1, families F1/F2, seeds 30–34.

The structure question here is about MATCH between prompt shape and task
shape: extraction prompts are inherently fielded, so if per-field structure
wins here while per-label loses on classification, structure's story becomes
shape-match, not universal benefit.

Usage:
  python mechanistic/e3_extraction.py --prep
  python mechanistic/e3_extraction.py --run free:f1:30 | --stream f1 | --stream f2
  python mechanistic/e3_extraction.py --analyze
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from e1_frontloading import FAMILIES  # noqa: E402

from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.lmutil import make_lm, reflection_callable, text_of  # noqa: E402

DATA_PATH = HERE.parent / "benchmarks" / "data" / "mit_restaurant.json"
RESULTS_DIR = HERE / "results_e3x"
RUNS_DIR = HERE / "runs_e3x"
SLOTS = ["Rating", "Amenity", "Location", "Restaurant_Name", "Price", "Hours",
         "Dish", "Cuisine"]
SEEDS = [30, 31, 32, 33, 34]
BUDGET = 800

LABEL2ID = {"O": 0, "B-Rating": 1, "I-Rating": 2, "B-Amenity": 3, "I-Amenity": 4,
            "B-Location": 5, "I-Location": 6, "B-Restaurant_Name": 7,
            "I-Restaurant_Name": 8, "B-Price": 9, "B-Hours": 10, "I-Hours": 11,
            "B-Dish": 12, "I-Dish": 13, "B-Cuisine": 14, "I-Price": 15, "I-Cuisine": 16}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

FIELD_DEFS = {
    "Rating": "star ratings or quality mentions (e.g. 'five star', 'best', 'highly rated')",
    "Amenity": "features/services (e.g. 'outdoor seating', 'wifi', 'kid friendly', 'delivery')",
    "Location": "places, streets, proximity phrases (e.g. 'near downtown', 'on 5th street')",
    "Restaurant_Name": "specific restaurant names",
    "Price": "price levels (e.g. 'cheap', 'expensive', 'under 20 dollars')",
    "Hours": "times/openness (e.g. 'open now', 'late night', 'for breakfast')",
    "Dish": "specific dishes or menu items",
    "Cuisine": "cuisine types (e.g. 'italian', 'sushi', 'bbq')",
}

FREE_SEED = (
    "Extract restaurant-search slots from the user query. The slot types are: "
    + ", ".join(SLOTS) + ". Return ONLY a JSON object mapping each slot type that occurs "
    'to a list of the exact text spans, e.g. {"Cuisine": ["italian"], "Price": ["cheap"]}. '
    "Use exact substrings of the query; omit slot types that do not occur."
)

# Factorial control: the SAME slot definitions as the codebook, but delivered
# as one free-text blob (not fielded components). Isolates DEFINITIONS from
# STRUCTURE in the 2x2 (structure: blob|fielded) x (definitions: absent|present).
FREE_SEED_WITH_DEFS = (
    "Extract restaurant-search slots from the user query. The slot types, with "
    "what each covers:\n"
    + "\n".join(f"- {name}: {d}" for name, d in FIELD_DEFS.items())
    + "\nReturn ONLY a JSON object mapping each slot type that occurs to a list of "
    'the exact text spans, e.g. {"Cuisine": ["italian"], "Price": ["cheap"]}. '
    "Use exact substrings of the query; omit slot types that do not occur."
)


def spans_from_bio(tokens: list[str], tags: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur_type, cur_toks = None, []
    for tok, tag_id in [*zip(tokens, tags, strict=True), ("", 0)]:
        lab = ID2LABEL.get(tag_id, "O") if tok != "" else "O"
        if lab.startswith("B-") or lab == "O" or (
                cur_type and lab.startswith("I-") and lab[2:] != cur_type):
            if cur_type:
                out.setdefault(cur_type, []).append(" ".join(cur_toks))
            cur_type, cur_toks = (lab[2:], [tok]) if lab.startswith("B-") else (None, [])
        elif lab.startswith("I-") and cur_type:
            cur_toks.append(tok)
    return out


def prep() -> None:
    from datasets import load_dataset

    base = ("https://huggingface.co/datasets/tner/mit_restaurant/resolve/"
            "refs%2Fconvert%2Fparquet/mit_restaurant/{split}/0000.parquet")

    def rows(split):
        ds = load_dataset("parquet", data_files={split: base.format(split=split)},
                          split=split)
        out = []
        for ex in ds:
            gold = spans_from_bio(ex["tokens"], ex["tags"])
            out.append({"text": " ".join(ex["tokens"]), "gold": gold})
        return out

    rng = random.Random(0)
    train_all = rows("train")
    test_all = rows("test")
    rng.shuffle(train_all)
    rng.shuffle(test_all)
    payload = {"train": train_all[:150], "dev": train_all[150:250], "test": test_all[:300]}
    DATA_PATH.write_text(json.dumps(payload, indent=0))
    print("mit_restaurant: 150/100/300 written")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def slot_f1(gold: dict, pred: dict) -> tuple[float, Counter, Counter, Counter]:
    """Per-example micro F1 over (type, normalized span) multisets."""
    g = Counter((t, norm(s)) for t, spans in gold.items() for s in spans)
    p = Counter((t, norm(s)) for t, spans in (pred or {}).items()
                if t in SLOTS for s in (spans if isinstance(spans, list) else [spans])
                if isinstance(s, str))
    tp = sum((g & p).values())
    prec = tp / max(1, sum(p.values()))
    rec = tp / max(1, sum(g.values()))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else (1.0 if not g and not p else 0.0)
    if not g and not p:
        f1 = 1.0
    return f1, g, p, g & p


def parse_json_obj(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def render_extraction(components: dict[str, str]) -> str:
    if "instruction" in components:
        return components["instruction"]
    parts = [components.get("task_instruction", "")]
    parts.append("Slot types:")
    for s in SLOTS:
        d = components.get(f"field::{s}", "").strip()
        parts.append(f"- {s}: {d}" if d else f"- {s}")
    br = components.get("boundary_rules", "").strip()
    if br:
        parts.append("Boundary rules:\n" + br)
    parts.append('Return ONLY a JSON object mapping occurring slot types to lists of exact '
                 'text spans from the query, e.g. {"Cuisine": ["italian"]}. '
                 "Omit slot types that do not occur.")
    return "\n\n".join(p for p in parts if p)


def extract_batch(lm, prompt: str, examples: list[dict], threads=8):
    from concurrent.futures import ThreadPoolExecutor

    def run(ex):
        try:
            raw = text_of(lm(messages=[{"role": "system", "content": prompt},
                                       {"role": "user", "content": ex["text"]}]))
        except Exception as e:  # noqa: BLE001
            raw = f"<error {e}>"
        pred = parse_json_obj(raw)
        f1, g, p, hit = slot_f1(ex["gold"], pred)
        missing = list((g - hit).keys())[:4]
        spurious = list((p - hit).keys())[:4]
        fb = (f"F1 {f1:.2f}." + (f" Missed: {missing}." if missing else "")
              + (f" Spurious: {spurious}." if spurious else "")
              + ("" if pred is not None else " Output was not valid JSON."))
        return {"text": ex["text"], "gold": ex["gold"], "raw": raw[-300:],
                "score": f1, "feedback": fb}

    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(run, examples))


class ExtractionAdapter:
    propose_new_texts = None  # default gepa proposer (single- or multi-component)

    def __init__(self, task_lm, valset):
        self.lm = task_lm
        self.metric_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        preds = extract_batch(self.lm, render_extraction(candidate), batch)
        self.metric_calls += len(batch)
        trajs = ([{"example": b, "prediction": p} for b, p in zip(batch, preds, strict=True)]
                 if capture_traces else None)
        return EvaluationBatch(outputs=[p["raw"] for p in preds],
                               scores=[p["score"] for p in preds], trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        out = {}
        trajs = eval_batch.trajectories or []
        worst = sorted((t["prediction"] for t in trajs), key=lambda p: p["score"])[:6]
        for comp in components_to_update:
            out[comp] = [{"Inputs": {"query": p["text"],
                                     "gold_slots": json.dumps(p["gold"])},
                          "Generated Outputs": {"response": p["raw"]},
                          "Feedback": p["feedback"]} for p in worst]
        return out


def codebook_seed() -> dict[str, str]:
    cand = {"task_instruction":
            "Extract restaurant-search slots from the user query as exact text spans."}
    for s in SLOTS:
        cand[f"field::{s}"] = FIELD_DEFS[s]
    cand["boundary_rules"] = ""
    return cand


def codebook_seed_nodefs() -> dict[str, str]:
    """Fielded structure with EMPTY per-field definitions (names only) — the
    structure-without-definitions cell of the factorial."""
    cand = {"task_instruction":
            "Extract restaurant-search slots from the user query as exact text spans."}
    for s2 in SLOTS:
        cand[f"field::{s2}"] = ""
    cand["boundary_rules"] = ""
    return cand


def run_cell(arm: str, family: str, seed: int) -> dict:
    out_path = RESULTS_DIR / f"{arm}_{family}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    import gepa

    data = json.loads(DATA_PATH.read_text())
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 200)
    refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    t0, r0 = len(task_lm.history), len(refl_lm.history)
    adapter = ExtractionAdapter(task_lm, data["dev"])
    seed_cand = {
        "free": {"instruction": FREE_SEED},
        "free_defs": {"instruction": FREE_SEED_WITH_DEFS},
        "codebook": codebook_seed(),
        "codebook_nodefs": codebook_seed_nodefs(),
    }[arm]
    run_dir = RUNS_DIR / f"{arm}_{family}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate=seed_cand, trainset=data["train"], valset=data["dev"],
        adapter=adapter, reflection_lm=reflection_callable(refl_lm),
        max_metric_calls=BUDGET, run_dir=str(run_dir), seed=seed,
        display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = dict(result.candidates[best_idx])
    test_preds = extract_batch(task_lm, render_extraction(best), data["test"])
    test_f1 = sum(p["score"] for p in test_preds) / len(test_preds)
    rec = {
        "arm": arm, "family": family, "seed": seed,
        "test_slot_f1": round(test_f1, 4),
        "seed_dev": round(scores[0], 4), "best_dev": round(scores[best_idx], 4),
        "n_components": len(best),
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(refl_lm, r0)), 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: test slot-F1 {test_f1:.3f} (${rec['cost_usd']:.2f})")
    return rec


def analyze() -> None:
    import math
    import statistics

    rows = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    for fam in ("f1", "f2"):
        for arm in ("free", "codebook"):
            rs = [r["test_slot_f1"] for r in rows if r["family"] == fam and r["arm"] == arm]
            if rs:
                ci = 1.96 * statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else 0
                print(f"{fam} {arm}: {statistics.mean(rs):.3f} ± {ci:.3f} (n={len(rs)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--run")  # arm:family:seed
    ap.add_argument("--stream", choices=("f1", "f2"))
    ap.add_argument("--factorial", choices=("f1", "f2"))
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.prep:
        prep()
    elif args.run:
        arm, fam, seed = args.run.split(":")
        run_cell(arm, fam, int(seed))
    elif args.stream:
        for arm in ("free", "codebook"):
            for s in SEEDS:
                run_cell(arm, args.stream, s)
    elif args.factorial:
        # the 2 NEW cells; existing free/codebook complete the 2x2
        for arm in ("free_defs", "codebook_nodefs"):
            for s in SEEDS:
                run_cell(arm, args.factorial, s)
    elif args.analyze:
        analyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
