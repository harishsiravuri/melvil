"""E5 Phase A — adaptive-budget stopping rules, simulated on logged curves ($0).

Rules (all computable EXACTLY from logged accepted-candidate curves — no
reconstruction of rejected-proposal counts, which curves cannot distinguish
from gepa's skip iterations):

  A(k): stop after the k-th accepted candidate (immediately post its full
        dev eval). Directly cashes in claim C1 (front-loading): if the first
        accepts carry the gain, A(1)/A(2) should retain most of it.
  G(C): stop once C metric calls pass without a new accepted candidate
        (C = 60, 120, 200, 300). The deployable patience rule.

Structural bound worth knowing: with dev=100, each accepted candidate costs
~106 calls of the 800 budget (full dev eval dominates), so even perfect rules
can only save the post-last-accept tail plus rejected-iteration overhead.

Outputs: mechanistic/results_sim/stopping.json + plots/e5_frontier.png
Run: python mechanistic/sim_stopping.py
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
TOTAL = 800
FULL_EVAL = 100  # a discovery at m is followed by its full eval ending ~m+100
MIN_GAIN = 0.005

ACCEPT_KS = (1, 2, 3)
GAP_CS = (60, 120, 200, 300)


def load_pool() -> list[dict]:
    return [json.loads(line) for line in (HERE / "curve_pool.jsonl").read_text().splitlines()]


def accepts(run: dict) -> list[tuple[int, float]]:
    """(spend_after_full_eval, running_best_dev) per accepted candidate,
    excluding the seed candidate (index 0)."""
    pts = sorted(run["curve"], key=lambda c: c["metric_calls"])
    out, best = [], pts[0]["dev"]
    for p in pts[1:]:
        best = max(best, p["dev"])
        out.append((min(p["metric_calls"] + FULL_EVAL, TOTAL), best))
    return out


def apply_accept_rule(run: dict, k: int) -> tuple[int, float]:
    acc = accepts(run)
    seed_dev = run["curve"][0]["dev"]
    if len(acc) >= k:
        return acc[k - 1]
    return (TOTAL, acc[-1][1] if acc else seed_dev)


def apply_gap_rule(run: dict, c: int) -> tuple[int, float]:
    acc = accepts(run)
    seed_dev = run["curve"][0]["dev"]
    last_accept_end = FULL_EVAL  # seed candidate's full eval
    best = seed_dev
    for spend, dev in acc:
        if spend - last_accept_end > c:  # patience expired before this accept
            return (min(last_accept_end + c, TOTAL), best)
        last_accept_end, best = spend, dev
    return (min(last_accept_end + c, TOTAL), best)


def ci95(vals):
    if len(vals) < 2:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def main() -> None:
    pool = load_pool()
    rules = [(f"A(k={k})", "A", k) for k in ACCEPT_KS] + [
        (f"G(C={c})", "G", c) for c in GAP_CS]
    out: dict = {"n_runs": len(pool), "rules": {}, "notes": [
        "A(k)/G(C) computed exactly from accepted-candidate curves",
        "budget frac relative to nominal 800; full evals dominate accept cost",
    ]}
    for label, kind, param in rules:
        fracs, rets = [], []
        n_flat = 0
        for run in pool:
            seed_dev = run["curve"][0]["dev"]
            acc = accepts(run)
            final_best = acc[-1][1] if acc else seed_dev
            gain = final_best - seed_dev
            spend, best = (apply_accept_rule(run, param) if kind == "A"
                           else apply_gap_rule(run, param))
            if gain < MIN_GAIN:
                n_flat += 1
                continue
            fracs.append(spend / TOTAL)
            rets.append((best - seed_dev) / gain)
        out["rules"][label] = {
            "mean_budget_frac": round(statistics.mean(fracs), 4),
            "budget_ci95": round(ci95(fracs), 4),
            "mean_retention": round(statistics.mean(rets), 4),
            "retention_ci95": round(ci95(rets), 4),
            "median_retention": round(statistics.median(rets), 4),
            "pct_full_retention": round(sum(1 for r in rets if r >= 0.999) / len(rets), 3),
            "n_scored": len(rets), "n_flat_excluded": n_flat,
        }
        s = out["rules"][label]
        print(f"{label}: budget {s['mean_budget_frac']:.0%}±{s['budget_ci95']:.0%} | "
              f"retention {s['mean_retention']:.1%}±{s['retention_ci95']:.1%} "
              f"(median {s['median_retention']:.0%}, full {s['pct_full_retention']:.0%}) "
              f"n={s['n_scored']}")

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "stopping.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    styles = {"A": ("#2a78d6", "o"), "G": ("#1baf7a", "s")}
    for label, s in out["rules"].items():
        color, marker = styles[label[0]]
        ax.errorbar(s["mean_budget_frac"] * 100, s["mean_retention"] * 100,
                    xerr=s["budget_ci95"] * 100, yerr=s["retention_ci95"] * 100,
                    marker=marker, ms=7, color=color, capsize=3, lw=1)
        ax.annotate(label, (s["mean_budget_frac"] * 100, s["mean_retention"] * 100),
                    xytext=(6, -4), textcoords="offset points", fontsize=8, color=color)
    ax.axhline(100, color="#888888", lw=0.7, ls=":")
    ax.axhline(95, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_xlabel("% of fixed budget spent")
    ax.set_ylabel("% of final dev gain retained")
    ax.set_title(f"E5-A stopping-rule frontier ({out['n_runs']} logged GEPA runs; "
                 "A(k)=stop after k accepts, G(C)=patience C calls)")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    fig.savefig(HERE / "plots" / "e5_frontier.png")
    print("wrote results_sim/stopping.json + plots/e5_frontier.png")


if __name__ == "__main__":
    main()
