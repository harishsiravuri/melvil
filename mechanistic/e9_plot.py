"""E9 figure — forest plot of the diagnosis-in-loop effect, both families.

Per-task seed-paired B - A (diagnosis-augmented GEPA minus vanilla GEPA at
matched budget) with 95% CIs, one panel per model family, plus the pooled
estimate. Reads results_sim/e9_full.json written by e9_diag_in_loop.py
--analyze.

    python mechanistic/e9_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SRC = HERE / "results_sim" / "e9_full.json"
OUT = HERE / "plots" / "e9_diagnosis_in_loop.png"

FAM_LABEL = {"f1": "GPT (gpt-4.1-mini / gpt-4.1)",
             "f2": "Claude (haiku-4.5 / sonnet-4.5)"}
ORDER = ["banking77", "clinc150", "massive", "trec", "ag_news",
         "emotion", "sst5", "stance_abortion"]


def main() -> None:
    data = json.loads(SRC.read_text())
    fams = [f for f in ("f1", "f2") if f in data["per_family"]]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 4.6),
                             sharex=True)
    if len(fams) == 1:
        axes = [axes]

    for ax, fam in zip(axes, fams, strict=False):
        fd = data["per_family"][fam]
        tasks = [t for t in ORDER if t in fd["per_task"]]
        rows = [(t, fd["per_task"][t]["B_minus_A"]) for t in tasks]
        pooled = fd["pooled_B_minus_A"]

        ys = list(range(len(rows)))[::-1]
        for y, (_task, st) in zip(ys, rows, strict=False):
            deg = st.get("degenerate")
            colour = {"W": "#1a7f37", "L": "#c1121f", "T": "#6b7280"}[st["verdict"]]
            ax.errorbar(st["mean"], y,
                        xerr=[[st["mean"] - st["lo"]], [st["hi"] - st["mean"]]],
                        fmt="o", ms=6, capsize=4, lw=1.8,
                        color=colour, mfc="white" if deg else colour, zorder=3)

        # pooled estimate, offset below the per-task rows
        py = -1.4
        ax.errorbar(pooled["mean"], py,
                    xerr=[[pooled["mean"] - pooled["lo"]],
                          [pooled["hi"] - pooled["mean"]]],
                    fmt="D", ms=8, capsize=5, lw=2.4, color="#111827", zorder=4)

        ax.axvline(0, color="#9ca3af", lw=1, ls="--", zorder=1)
        ax.axhline(-0.7, color="#d1d5db", lw=0.8, zorder=1)
        ax.set_yticks(ys + [py])
        ax.set_yticklabels([t for t, _ in rows]
                           + [f"POOLED (n={fd['n_pooled']})"],
                           fontsize=9)
        for lbl in ax.get_yticklabels()[-1:]:
            lbl.set_fontweight("bold")
        ax.set_ylim(py - 0.8, len(rows) - 0.4)
        w = fd["wtl_B_vs_A"]
        ax.set_title(f"{FAM_LABEL[fam]}\nW/T/L {len(w['W'])}/{len(w['T'])}/{len(w['L'])}",
                     fontsize=10)
        ax.set_xlabel("accuracy difference  (diagnosis-in-loop − vanilla GEPA)",
                      fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("E9 — injecting the aggregated diagnosis into GEPA's reflection, "
                 "at matched budget (B1=800, seeds 20–24)", fontsize=11)
    fig.text(0.5, 0.005,
             "green = 95% CI above 0 · red = CI below 0 · grey = tie · "
             "hollow = saturated task (zero-variance CI, not scored)",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
