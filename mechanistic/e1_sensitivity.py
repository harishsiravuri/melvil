"""E1 (C1) sensitivity analyses + early-to-final rank correlation.

Robustness of the "first accepted proposal captures a median 0.33 of the gain"
claim under alternative definitions, plus the task-stratified breakdown,
per-cell exclusion counts, and the Spearman rank correlation between
early-checkpoint and final dev standing (the mechanism behind E6's dead racing).

Definitions (all from GEPA per-candidate dev curves; candidate 0 = seed, later
candidates = accepted proposals in discovery order):
  V1 running-max share   : (runmax after accept 1 - seed)/(final runmax - seed)
  V2 absolute first gain : runmax-increment of accept 1, in accuracy points
  V3 first-improver share: among accepts that RAISED the running max, the share
                           captured by the first such accept (handles the case
                           where accept 1 was a lateral/negative full-val move)

Run: python mechanistic/e1_sensitivity.py
Outputs: results_sim/e1_sensitivity.json + prints tables.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_E1 = HERE / "results_e1"
POOL = HERE / "curve_pool.jsonl"
MIN_GAIN = 0.005
TASK_TYPE = {"banking77": "classification", "trec": "classification",
             "massive": "classification", "bbh_geometric": "bbh-reasoning",
             "gsm8k": "math-qa"}
RACE_FS = [0.15, 0.25, 0.40]
T = 800


def increments(curve: list[dict]) -> tuple[float, list[float]]:
    """(seed_dev, running-max increments per accepted candidate)."""
    pts = sorted(curve, key=lambda c: c["metric_calls"])
    seed = pts[0]["dev"]
    best, incs = seed, []
    for p in pts[1:]:
        new = max(best, p["dev"])
        incs.append(new - best)
        best = new
    return seed, incs


def spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da and db else None


def ci95(v):
    return 1.96 * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def sensitivity() -> dict:
    rows = [json.loads(p.read_text()) for p in RESULTS_E1.glob("gepa_*.json")]
    cells: dict = {}
    excl: dict = {}
    for r in rows:
        key = (r["task"], r["budget"], r["family"])
        cells.setdefault(key, {"v1": [], "v2": [], "v3": []})
        excl.setdefault(key, {"total": 0, "excluded_flat": 0})
        excl[key]["total"] += 1
        seed, incs = increments(r["curve"])
        total = sum(incs)
        if total < MIN_GAIN or not incs:
            excl[key]["excluded_flat"] += 1
            continue
        cells[key]["v1"].append(incs[0] / total)
        cells[key]["v2"].append(incs[0])
        pos = [i for i, v in enumerate(incs) if v > 1e-9]
        if pos:
            first_improver = pos[0]
            cells[key]["v3"].append(incs[first_improver] / total)

    def agg(vals):
        return {"median": round(statistics.median(vals), 3),
                "mean": round(statistics.mean(vals), 3),
                "ci95": round(ci95(vals), 3), "n": len(vals)} if vals else None

    out = {"per_cell": {}, "pooled": {}, "by_task_type": {}, "exclusions": {}}
    pool_v = {"v1": [], "v2": [], "v3": []}
    tt_v: dict = {}
    for key, d in cells.items():
        sk = "|".join(map(str, key))
        out["per_cell"][sk] = {k: agg(v) for k, v in d.items()}
        out["exclusions"][sk] = excl[key]
        # pool F1 B1 for the headline, keep all for type strat
        if key[2] == "f1":
            for k in ("v1", "v2", "v3"):
                pool_v[k].extend(d[k])
        tt = TASK_TYPE.get(key[0], "classification")
        tt_v.setdefault(tt, {"v1": [], "v2": [], "v3": []})
        for k in ("v1", "v2", "v3"):
            tt_v[tt][k].extend(d[k])
    out["pooled"] = {k: agg(v) for k, v in pool_v.items()}
    out["by_task_type"] = {tt: {k: agg(v) for k, v in d.items()} for tt, d in tt_v.items()}
    return out


def rank_correlation() -> dict:
    groups: dict = {}
    for line in POOL.read_text().splitlines():
        r = json.loads(line)
        if not (r["arm"] == "vanilla" or r["arm"] == "vanilla_b800"):
            continue
        g = groups.setdefault(r["task"], {})
        g.setdefault(r["seed"], r["curve"])  # dedupe by seed
    out = {"per_F": {}, "note": "Spearman(dev@F*T, dev@T) across seeds, per task, mean"}

    def dev_at(curve, t):
        best = None
        for p in sorted(curve, key=lambda c: c["metric_calls"]):
            if p["metric_calls"] <= t:
                best = p["dev"] if best is None else max(best, p["dev"])
        return best if best is not None else curve[0]["dev"]

    for f in RACE_FS:
        rhos = []
        for _task, seedmap in groups.items():
            if len(seedmap) < 3:
                continue
            curves = list(seedmap.values())
            early = [dev_at(c, f * T) for c in curves]
            final = [dev_at(c, T) for c in curves]
            rho = spearman(early, final)
            if rho is not None:
                rhos.append(rho)
        out["per_F"][f] = {"mean_spearman": round(statistics.mean(rhos), 3),
                           "ci95": round(ci95(rhos), 3), "n_tasks": len(rhos)}
    return out


def main() -> None:
    sens = sensitivity()
    rank = rank_correlation()
    out = {"sensitivity": sens, "rank_correlation": rank}
    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "e1_sensitivity.json").write_text(json.dumps(out, indent=1))

    print("=== first-accept share, pooled F1 (all budgets/tasks) ===")
    for k, label in [("v1", "V1 running-max share"), ("v2", "V2 absolute gain (pts)"),
                     ("v3", "V3 first-improver share")]:
        a = sens["pooled"][k]
        print(f"  {label:<26} median {a['median']:.3f}  mean {a['mean']:.3f} "
              f"± {a['ci95']:.3f}  (n={a['n']})")
    print("\n=== by task type (V1 running-max share) ===")
    for tt, d in sens["by_task_type"].items():
        a = d["v1"]
        if a:
            print(f"  {tt:<18} median {a['median']:.3f} mean {a['mean']:.3f} (n={a['n']})")
    print("\n=== exclusions (flat runs, total gain < 0.005) ===")
    n_excl = sum(v["excluded_flat"] for v in sens["exclusions"].values())
    n_tot = sum(v["total"] for v in sens["exclusions"].values())
    print(f"  {n_excl}/{n_tot} runs excluded as flat across all cells")
    print("\n=== early-to-final rank correlation (racing pool) ===")
    for f, d in rank["per_F"].items():
        print(f"  F={f:.2f}: mean Spearman {d['mean_spearman']:+.3f} "
              f"± {d['ci95']:.3f} (n_tasks={d['n_tasks']})")
    print("\nwrote results_sim/e1_sensitivity.json")


if __name__ == "__main__":
    main()
