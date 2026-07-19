"""Aggregate results_confirm/ into the headline table + plot.

Run: python benchmarks/make_confirmation_report.py
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
from run_confirmation import CONFIRM_SEEDS, CONFIRM_TASKS  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results_confirm"
PLOTS_DIR = Path(__file__).parent / "plots"

ARM_ORDER = ["seed", "vanilla", "full", "miprov2"]
ARM_LABELS = {
    "seed": "seed prompt",
    "vanilla": "vanilla GEPA",
    "full": "melvil full (v0.2)",
    "miprov2": "MIPROv2 (dspy-native)",
}
ARM_COLORS = {"seed": "#8a8a86", "vanilla": "#2a78d6", "full": "#1baf7a", "miprov2": "#4a3aa7"}


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]


def rows_for(rs, task, arm):
    return [r for r in rs if r["task"] == task and r["arm"] == arm]


def table(rs) -> list[str]:
    lines = [
        "| arm | " + " | ".join(CONFIRM_TASKS) + " | mean |",
        "|---|" + "---|" * (len(CONFIRM_TASKS) + 1),
    ]
    for metric, name in [("test_accuracy", "accuracy"), ("test_macro_f1", "macro-F1")]:
        lines.append(f"| **{name}** |" + " |" * (len(CONFIRM_TASKS) + 1))
        for arm in ARM_ORDER:
            cells, means = [], []
            for task in CONFIRM_TASKS:
                rows = rows_for(rs, task, arm)
                if rows:
                    m = statistics.mean(r[metric] for r in rows)
                    cells.append(f"{m:.3f}")
                    means.append(m)
                else:
                    cells.append("—")
            complete = len(means) == len(CONFIRM_TASKS)
            overall = f"**{statistics.mean(means):.3f}**" if complete else "—"
            lines.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + f" | {overall} |")
    return lines


def plot(rs) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5), dpi=150)
    width = 0.8 / len(ARM_ORDER)
    for ai, arm in enumerate(ARM_ORDER):
        xs, ys, lo, hi = [], [], [], []
        for ti, task in enumerate(CONFIRM_TASKS):
            rows = rows_for(rs, task, arm)
            if not rows:
                continue
            vals = [r["test_accuracy"] for r in rows]
            xs.append(ti + (ai - len(ARM_ORDER) / 2 + 0.5) * width)
            ys.append(statistics.mean(vals))
            lo.append(statistics.mean(vals) - min(vals))
            hi.append(max(vals) - statistics.mean(vals))
        ax.bar(xs, ys, width * 0.9, color=ARM_COLORS[arm], label=ARM_LABELS[arm],
               yerr=[lo, hi] if any(lo) or any(hi) else None,
               error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#555555"})
    ax.set_xticks(range(len(CONFIRM_TASKS)), CONFIRM_TASKS, fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(f"Confirmation pass: test accuracy (fresh seeds {CONFIRM_SEEDS}, "
                 "budget=light; whiskers = min–max)")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / "confirmation_test_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    rs = load()
    done = {(r["task"], r["arm"], r["seed"]) for r in rs}
    expected = {(t, a, s) for t in CONFIRM_TASKS for a in ARM_ORDER[1:] for s in CONFIRM_SEEDS}
    expected |= {(t, "seed", CONFIRM_SEEDS[0]) for t in CONFIRM_TASKS}
    missing = sorted(expected - done)
    if missing:
        print(f"WARNING: {len(missing)} runs missing, e.g. {missing[:6]}")
    lines = table(rs)
    total = sum(r.get("optimize_cost_usd", 0) + r.get("test_cost_usd", 0) for r in rs)
    lines += ["", f"Confirmation-pass spend: **${total:.2f}**"]
    out = plot(rs)
    lines += ["", f"![confirmation](plots/{out.name})"]
    md = Path(__file__).parent / "confirmation_tables.md"
    md.write_text("\n".join(lines))
    print("\n".join(lines[:20]))
    print(f"wrote {md} + {out}")


if __name__ == "__main__":
    main()
