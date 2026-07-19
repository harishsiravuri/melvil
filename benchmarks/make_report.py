"""Aggregate benchmarks/results/*.json into RESULTS.md tables + plots.

Run: python benchmarks/make_report.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from prep_data import TASKS  # noqa: E402
from run_matrix import ARMS, SEEDS, TASK_MODEL, TRANSFER_MODEL  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

ARM_ORDER = ["seed", "vanilla", "full", "no_codebook", "no_confusion", "no_mining"]
ARM_LABELS = {
    "seed": "seed prompt",
    "vanilla": "vanilla GEPA",
    "full": "melvil full",
    "no_codebook": "full − codebook",
    "no_confusion": "full − confusion",
    "no_mining": "full − mining",
}
# categorical palette (validated, light surface); seed baseline gets neutral gray
ARM_COLORS = {
    "seed": "#8a8a86",
    "vanilla": "#2a78d6",
    "full": "#1baf7a",
    "no_codebook": "#eda100",
    "no_confusion": "#4a3aa7",
    "no_mining": "#e34948",
}


def load_results() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]


def rows_for(results: list[dict], task: str, arm: str) -> list[dict]:
    return [r for r in results if r["task"] == task and r["arm"] == arm]


def mean_range(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} [{min(vals):.3f}–{max(vals):.3f}]"


def task_table(results: list[dict], task: str) -> list[str]:
    lines = [
        f"### {task}",
        "",
        "| arm | test accuracy (3 seeds) | test macro-F1 | dev acc | cost/run |",
        "|---|---|---|---|---|",
    ]
    for arm in ARM_ORDER:
        rows = rows_for(results, task, arm)
        if not rows:
            continue
        acc = [r["test_accuracy"] for r in rows]
        f1 = [r["test_macro_f1"] for r in rows]
        dev = [r["dev_accuracy"] for r in rows if r.get("dev_accuracy") is not None]
        cost = [r.get("optimize_cost_usd", 0) + r.get("test_cost_usd", 0) for r in rows]
        lines.append(
            f"| {ARM_LABELS[arm]} | {mean_range(acc)} | {mean_range(f1)} | "
            f"{mean_range(dev)} | ${statistics.mean(cost):.2f} |"
        )
    tr = rows_for(results, task, "transfer")
    if tr:
        acc = [r["test_accuracy"] for r in tr]
        f1 = [r["test_macro_f1"] for r in tr]
        lines.append(
            f"| full → `{TRANSFER_MODEL.split('/')[-1]}` (transfer) | {mean_range(acc)} | "
            f"{mean_range(f1)} | — | ${statistics.mean(r['test_cost_usd'] for r in tr):.2f} |"
        )
    lines.append("")
    return lines


def overall_table(results: list[dict]) -> list[str]:
    lines = [
        "## Overall (mean of task means, test accuracy)",
        "",
        "| arm | " + " | ".join(TASKS) + " | mean |",
        "|---|" + "---|" * (len(TASKS) + 1),
    ]
    for arm in ARM_ORDER:
        cells, means = [], []
        for task in TASKS:
            rows = rows_for(results, task, arm)
            if rows:
                m = statistics.mean(r["test_accuracy"] for r in rows)
                cells.append(f"{m:.3f}")
                means.append(m)
            else:
                cells.append("—")
        overall = f"**{statistics.mean(means):.3f}**" if len(means) == len(TASKS) else "—"
        lines.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + f" | {overall} |")
    lines.append("")
    return lines


def bar_plot(results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    n_arms = len(ARM_ORDER)
    width = 0.8 / n_arms
    for ai, arm in enumerate(ARM_ORDER):
        xs, ys, lo, hi = [], [], [], []
        for ti, task in enumerate(TASKS):
            rows = rows_for(results, task, arm)
            if not rows:
                continue
            vals = [r["test_accuracy"] for r in rows]
            xs.append(ti + (ai - n_arms / 2 + 0.5) * width)
            ys.append(statistics.mean(vals))
            lo.append(statistics.mean(vals) - min(vals))
            hi.append(max(vals) - statistics.mean(vals))
        ax.bar(xs, ys, width * 0.92, color=ARM_COLORS[arm], label=ARM_LABELS[arm],
               yerr=[lo, hi] if any(lo) or any(hi) else None,
               error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#555555"})
    ax.set_xticks(range(len(TASKS)), TASKS)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"Test accuracy by task and arm ({TASK_MODEL.split('/')[-1]}, "
                 f"budget=light, mean of {len(SEEDS)} seeds; whiskers = min–max)")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / "test_accuracy_by_arm.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def curve_plot(results: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150, sharey=False)
    grid = list(range(0, 801, 10))
    for ti, task in enumerate(TASKS):
        ax = axes[ti // 3][ti % 3]
        for arm in ["vanilla", "full"]:
            rows = rows_for(results, task, arm)
            curves = []
            for r in rows:
                pts = sorted((c["metric_calls"], c["dev_score"]) for c in r.get("curve", []))
                best, i, vals = None, 0, []
                for t in grid:
                    while i < len(pts) and pts[i][0] <= t:
                        best = pts[i][1] if best is None else max(best, pts[i][1])
                        i += 1
                    vals.append(best)
                first = next((v for v in vals if v is not None), 0)
                curves.append([first if v is None else v for v in vals])
            if not curves:
                continue
            mean_cv = [statistics.mean(c[i] for c in curves) for i in range(len(grid))]
            for cv in curves:
                ax.plot(grid, cv, color=ARM_COLORS[arm], alpha=0.2, lw=0.8)
            ax.plot(grid, mean_cv, color=ARM_COLORS[arm], lw=2,
                    label=ARM_LABELS[arm] if ti == 0 else None)
        ax.set_title(task, fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1][0].set_xlabel("metric calls")
    axes[0][0].set_ylabel("best dev accuracy")
    axes[1][0].set_ylabel("best dev accuracy")
    fig.legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Dev accuracy vs budget: vanilla GEPA vs melvil full (thin = per seed)")
    fig.tight_layout()
    out = PLOTS_DIR / "dev_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def totals(results: list[dict]) -> str:
    total = sum(r.get("optimize_cost_usd", 0) + r.get("test_cost_usd", 0) for r in results)
    return f"Total measured spend for the matrix: **${total:.2f}**"


def main() -> None:
    results = load_results()
    done = {(r["task"], r["arm"], r.get("seed")) for r in results}
    expected = {(t, a, s) for t in TASKS for a in ARMS if a != "seed" for s in SEEDS} | {
        (t, "seed", 0) for t in TASKS
    }
    missing = expected - done
    lines = ["<!-- generated by benchmarks/make_report.py -->", ""]
    if missing:
        lines.append(f"**WARNING: {len(missing)} runs missing:** {sorted(missing)[:8]}...")
        lines.append("")
    lines += overall_table(results)
    for task in TASKS:
        lines += task_table(results, task)
    bar = bar_plot(results)
    cv = curve_plot(results)
    lines += [
        f"![test accuracy](plots/{bar.name})", "",
        f"![dev curves](plots/{cv.name})", "",
        totals(results), "",
    ]
    out = Path(__file__).parent / "results_tables.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} + {bar} + {cv}")
    print(totals(results))


if __name__ == "__main__":
    main()
