"""E6 Phase A — GEPA-Race simulation on logged multi-seed curves ($0).

Racing whole optimization trajectories: start N seeds, at checkpoint F% of the
total budget keep only the current dev leader. Two accountings:

- matched-total: total metric calls equal one full run T. The winner therefore
  only reaches t_w = T * (1 - (N-1)*F) of its own curve. (Combos where
  t_w < F*T are infeasible and skipped.) Dev-based comparison only — test
  scores don't exist for truncated runs.
- race-to-full: winner continues to its full T (total spend = T*(1+(N-1)*F)).
  Both dev and TEST comparisons available (winner's logged test score).

Baselines per seed-subset (the fair-skeptic set): the mean and the MAX
("luckiest seed") of the same N seeds' full-budget results.

Positioning note for the paper: CAPO (arXiv:2504.16005) and best-arm-
identification race candidate PROMPTS within one run; this races whole
optimization TRAJECTORIES.

Outputs: mechanistic/results_sim/racing.json + plots/e6_frontier.png
Run: python mechanistic/sim_racing.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
T = 800
NS = (2, 3, 4, 6)
FS = (0.15, 0.25, 0.40)
MAX_SUBSETS = 20
ARM = "vanilla"  # race the confirmed-strongest configuration


def load_pool() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for line in (HERE / "curve_pool.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == ARM:
            runs.setdefault(r["task"], []).append(r)
    for task, rs in runs.items():
        # dedupe (source, seed) collisions: keep first
        seen, uniq = set(), []
        for r in rs:
            if r["seed"] not in seen:
                seen.add(r["seed"])
                uniq.append(r)
        runs[task] = uniq
    return runs


def dev_at(run: dict, t: float) -> float:
    best = None
    for p in sorted(run["curve"], key=lambda c: c["metric_calls"]):
        if p["metric_calls"] <= t:
            best = p["dev"] if best is None else max(best, p["dev"])
    return best if best is not None else run["curve"][0]["dev"]


def ci95(vals):
    if len(vals) < 2:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def main() -> None:
    pool = load_pool()
    rng = random.Random(0)
    grid: dict = {"arm": ARM, "T": T, "cells": []}
    for n in NS:
        for f in FS:
            t_matched = T * (1 - (n - 1) * f)
            feasible = t_matched >= f * T
            d_mean_m, d_mean_full, d_lucky_full, dt_mean, dt_lucky = [], [], [], [], []
            n_subsets = 0
            for runs in pool.values():
                if len(runs) < n:
                    continue
                subsets = list(itertools.combinations(range(len(runs)), n))
                if len(subsets) > MAX_SUBSETS:
                    subsets = rng.sample(subsets, MAX_SUBSETS)
                for idxs in subsets:
                    group = [runs[i] for i in idxs]
                    winner = max(group, key=lambda r: dev_at(r, f * T))
                    full_devs = [dev_at(r, T) for r in group]
                    if feasible:
                        d_mean_m.append(dev_at(winner, t_matched) - statistics.mean(full_devs))
                    d_mean_full.append(dev_at(winner, T) - statistics.mean(full_devs))
                    d_lucky_full.append(dev_at(winner, T) - max(full_devs))
                    tests = [r["test_accuracy"] for r in group if r.get("test_accuracy")]
                    if winner.get("test_accuracy") and tests:
                        dt_mean.append(winner["test_accuracy"] - statistics.mean(tests))
                        dt_lucky.append(winner["test_accuracy"] - max(tests))
                    n_subsets += 1
            if not n_subsets:
                continue
            cell = {
                "N": n, "F": f, "n_subsets": n_subsets,
                "matched_total_feasible": feasible,
                "winner_budget_frac_matched": round(1 - (n - 1) * f, 3),
                "extra_budget_race_to_full": round(1 + (n - 1) * f, 3),
                "delta_dev_matched_vs_mean": (
                    round(statistics.mean(d_mean_m), 4) if d_mean_m else None),
                "delta_dev_matched_ci": round(ci95(d_mean_m), 4) if d_mean_m else None,
                "delta_dev_full_vs_mean": round(statistics.mean(d_mean_full), 4),
                "delta_dev_full_vs_lucky": round(statistics.mean(d_lucky_full), 4),
                "delta_test_full_vs_mean": round(statistics.mean(dt_mean), 4) if dt_mean else None,
                "delta_test_full_vs_mean_ci": round(ci95(dt_mean), 4) if dt_mean else None,
                "delta_test_full_vs_lucky": (
                    round(statistics.mean(dt_lucky), 4) if dt_lucky else None),
            }
            grid["cells"].append(cell)
            m = cell["delta_dev_matched_vs_mean"]
            print(f"N={n} F={f:.0%}: matched-dev {'+' if m and m>=0 else ''}{m} | "
                  f"full-dev vs mean {cell['delta_dev_full_vs_mean']:+.4f} "
                  f"vs lucky {cell['delta_dev_full_vs_lucky']:+.4f} | "
                  f"full-TEST vs mean {cell['delta_test_full_vs_mean']:+.4f} "
                  f"(+/-{cell['delta_test_full_vs_mean_ci']}) "
                  f"vs lucky {cell['delta_test_full_vs_lucky']:+.4f} | n={cell['n_subsets']}")

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "racing.json").write_text(json.dumps(grid, indent=1))

    # frontier: matched-total dev delta (primary), one line per N
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    colors = {2: "#2a78d6", 3: "#1baf7a", 4: "#eda100", 6: "#4a3aa7"}
    for metric, ax, title in [
        ("delta_dev_matched_vs_mean", axes[0],
         "matched TOTAL budget (winner truncated)"),
        ("delta_test_full_vs_mean", axes[1],
         "race-to-full (extra budget, TEST delta)"),
    ]:
        for n in NS:
            xs = [c["F"] * 100 for c in grid["cells"] if c["N"] == n and c[metric] is not None]
            ys = [c[metric] * 100 for c in grid["cells"] if c["N"] == n and c[metric] is not None]
            if xs:
                ax.plot(xs, ys, marker="o", color=colors[n], label=f"N={n}", lw=1.5)
        ax.axhline(0, color="#888888", lw=0.8, ls=":")
        ax.set_xlabel("checkpoint F (% of budget)")
        ax.set_ylabel("accuracy points vs mean single run")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("E6-A: GEPA-Race simulation (vanilla arm, logged multi-seed curves)")
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    fig.savefig(HERE / "plots" / "e6_frontier.png")
    print("wrote results_sim/racing.json + plots/e6_frontier.png")


if __name__ == "__main__":
    main()
