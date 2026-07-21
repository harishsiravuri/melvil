"""E1 analysis: cumulative fraction-of-total-gain vs accepted-proposal index,
per task x budget (GEPA; MIPROv2 from trial_logs where parseable), plus the
quantified front-loading verdict.

Run: python mechanistic/e1_analysis.py
Outputs: results_sim/e1_summary.json, plots/e1_cumulative_gain.png
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results_e1"
MIN_GAIN = 0.005
KMAX = 8  # plot up to the 8th accepted proposal
TASKS = ["banking77", "trec", "massive", "bbh_geometric", "gsm8k"]
BUDGETS = [800, 1600, 3200]
BUDGET_COLORS = {800: "#2a78d6", 1600: "#eda100", 3200: "#e34948"}


def gepa_cumfracs(rec: dict) -> list[float] | None:
    pts = sorted(rec["curve"], key=lambda c: c["metric_calls"])
    seed_dev = pts[0]["dev"]
    best = seed_dev
    increments = []
    for p in pts[1:]:
        new = max(best, p["dev"])
        increments.append(new - best)
        best = new
    total = best - seed_dev
    if total < MIN_GAIN:
        return None
    cum, out = 0.0, []
    for inc in increments:
        cum += inc
        out.append(cum / total)
    return out


def mipro_cumfracs(rec: dict) -> list[float] | None:
    """Best-so-far dev per trial from MIPROv2 trial_logs (defensive parsing)."""
    logs = rec.get("trial_logs")
    if not logs:
        return None
    try:
        items = sorted(((int(k), v) for k, v in logs.items()), key=lambda kv: kv[0])
        scores = []
        for _, v in items:
            s = v.get("full_eval_score", v.get("score"))
            if s is not None:
                scores.append(float(s))
        if len(scores) < 2:
            return None
        seed = scores[0]
        best, increments = seed, []
        for s in scores[1:]:
            new = max(best, s)
            if new > best:
                increments.append(new - best)
            best = new
        total = best - seed
        if total < MIN_GAIN or not increments:
            return None
        cum, out = 0.0, []
        for inc in increments:
            cum += inc
            out.append(cum / total)
        return out
    except Exception:  # noqa: BLE001 - unknown log schema -> skip run
        return None


def pad(fracs: list[float], k: int) -> list[float]:
    return [fracs[i] if i < len(fracs) else 1.0 for i in range(k)]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    summary: dict = {"per_cell": {}, "flat_runs": 0}
    curves: dict = {}  # (optimizer, task, budget, family) -> list of padded cumfrac lists
    for r in rows:
        fr = gepa_cumfracs(r) if r["optimizer"] == "gepa" else mipro_cumfracs(r)
        if fr is None:
            summary["flat_runs"] += 1
            continue
        key = (r["optimizer"], r["task"], r["budget"], r["family"])
        curves.setdefault(key, []).append(fr)

    for key, frs in sorted(curves.items()):
        first = [f[0] for f in frs]
        by2 = [f[1] if len(f) > 1 else 1.0 for f in frs]
        summary["per_cell"]["|".join(map(str, key))] = {
            "n": len(frs),
            "median_first": round(statistics.median(first), 3),
            "median_by_2": round(statistics.median(by2), 3),
            "median_n_accepts": statistics.median(len(f) for f in frs),
        }

    # headline numbers: gepa f1, pooled
    for fam in ("f1", "f2"):
        pool_first = [f[0] for (o, t, b, fm), frs in curves.items() if o == "gepa" and fm == fam
                      for f in frs]
        pool_by2 = [(f[1] if len(f) > 1 else 1.0) for (o, t, b, fm), frs in curves.items()
                    if o == "gepa" and fm == fam for f in frs]
        if pool_first:
            summary[f"gepa_{fam}_median_first"] = round(statistics.median(pool_first), 3)
            summary[f"gepa_{fam}_median_by2"] = round(statistics.median(pool_by2), 3)
            summary[f"gepa_{fam}_n"] = len(pool_first)
    mip_first = [f[0] for (o, *_), frs in curves.items() if o == "miprov2" for f in frs]
    if mip_first:
        summary["miprov2_median_first"] = round(statistics.median(mip_first), 3)
        summary["miprov2_n"] = len(mip_first)

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "e1_summary.json").write_text(json.dumps(summary, indent=1))

    # figure: one panel per task, x = accepted index, y = mean cumulative fraction
    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.1 * len(TASKS), 3.4),
                             dpi=150, sharey=True)
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        for budget in BUDGETS:
            frs = curves.get(("gepa", task, budget, "f1"), [])
            if not frs:
                continue
            padded = [pad(f, KMAX) for f in frs]
            mean_cv = [statistics.mean(p[i] for p in padded) for i in range(KMAX)]
            lo = [mean_cv[i] - 1.96 * statistics.stdev([p[i] for p in padded])
                  / math.sqrt(len(padded)) if len(padded) > 1 else mean_cv[i]
                  for i in range(KMAX)]
            hi = [2 * mean_cv[i] - lo[i] for i in range(KMAX)]
            xs = list(range(1, KMAX + 1))
            ax.plot(xs, mean_cv, marker="o", ms=3, lw=1.6, color=BUDGET_COLORS[budget],
                    label=f"B={budget}" if ti == 0 else None)
            ax.fill_between(xs, lo, hi, color=BUDGET_COLORS[budget], alpha=0.12, lw=0)
        ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("accepted proposal #")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("cumulative fraction of total dev gain")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("E1: gain concentration by accepted proposal (GEPA, F1; shading = 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    out = HERE / "plots" / "e1_cumulative_gain.png"
    fig.savefig(out)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cell"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
