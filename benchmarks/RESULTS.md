# Benchmark results — labelsmith v0.1.0

**Headline (honest version): the per-label codebook is the feature that matters.**
Averaged over six public classification tasks (3 seeds each, matched "light" budgets),
labelsmith's full classification layer reaches **0.829** mean test accuracy vs **0.817**
for vanilla GEPA and **0.762** for the unoptimized seed prompt. The ablations attribute
essentially all of that margin to the per-label codebook (removing it: **0.811**, the
largest single drop, −1.8 points); confusion-driven reflection is *neutral-to-slightly-
negative on average* in this v0.1 implementation (removing it: **0.832**), and
hard-example mining is roughly neutral (removing it: **0.826**). We report this as
measured — see "What we'd change" below.

Setup: task model `gpt-4.1-mini`, reflection `gpt-4.1` (via OpenRouter); budget preset
"light" (800 metric calls) everywhere; splits train 150 / dev 100 / test 300, stratified,
data seed 0; dev drives all optimization decisions; each task's test split touched exactly
once per (arm, seed), at the end. Arms and metrics were fixed before any results (see
`run_matrix.py`, committed before the launch). MIPROv2 (optional arm e) was not run in
v0.1 — it optimizes under dspy's chat-adapter format rather than our deployable rendered
prompt, so a clean comparison needs dedicated wiring; recorded here as future work.

## Overall (mean of per-task means, test accuracy)

| arm | banking77 | ag_news | emotion | trec | clinc150 | massive | mean |
|---|---|---|---|---|---|---|---|
| seed prompt | 0.760 | 0.830 | 0.563 | 0.600 | 0.980 | 0.840 | **0.762** |
| vanilla GEPA | 0.806 | 0.859 | 0.544 | 0.862 | 0.980 | 0.849 | **0.817** |
| labelsmith full | 0.824 | 0.844 | 0.562 | 0.881 | 0.982 | 0.881 | **0.829** |
| full − codebook | 0.773 | 0.837 | 0.560 | 0.840 | 0.987 | 0.870 | **0.811** |
| full − confusion | 0.827 | 0.886 | 0.564 | 0.860 | 0.983 | 0.871 | **0.832** |
| full − mining | 0.808 | 0.850 | 0.564 | 0.890 | 0.979 | 0.863 | **0.826** |

![test accuracy by task and arm](plots/test_accuracy_by_arm.png)

![dev curves](plots/dev_curves.png)

Full per-task tables (per-seed ranges, macro-F1, dev accuracy, per-run cost) are in
[results_tables.md](results_tables.md); raw per-run JSON in `results/`; every artifact and
its reflection traces in `runs/`.

## Reading the results

1. **Where optimization has headroom, the full layer wins on test.** On the three tasks
   with both headroom and label structure — banking77 (+1.9 over vanilla), trec (+1.9),
   MASSIVE (+3.2) — the full layer beats vanilla GEPA on test accuracy in every case. The
   dev-vs-budget curves (plot above) are comparable between the two arms (vanilla slightly
   ahead on banking77/ag_news dev, full ahead on massive/clinc150): the layer's edge shows
   up at test time, not as faster dev climbing.
2. **The codebook is the load-bearing feature.** `full − codebook` is the worst optimized
   arm on 4/6 tasks and costs −5.1 points on banking77 (25 labels) vs full. Structured
   per-label definitions are also what transfers (see 5): the prompt carries explicit
   label semantics rather than model-idiosyncratic phrasing.
3. **Confusion-driven reflection did not earn its keep (yet).** Removing it is +0.3 on
   average and +4.2 on ag_news. The traces show why: the confusion-driven component
   selector concentrates updates on the two most-confused labels and starves the rest —
   in the ag_news full run (seed 0), 36 of 42 component updates went to `Sci/Tech` and
   `Business` while `World` and `Sports` received zero. Focused boundary-sharpening is the
   right instinct with 25–30 labels, but on small taxonomies it over-commits. An honest
   negative for our headline feature as currently implemented.
4. **Mining is neutral on average and its dev gate is too permissive.** 53/54
   mining-enabled runs kept their exemplars, and on the headroom tasks the full arm's dev
   accuracy overstates test by +2.6 points on average (vanilla: −0.4; `full − mining`:
   −1.1) — dev-drawn exemplars inflate the dev score exactly as the documented caveat
   predicts, and persistently-missed dev examples select for label noise (clearest on AG
   News). trec is the one task where dropping mining *helped* (+0.9). Judge artifacts on
   test.
5. **Transfer to `gpt-4.1-nano` mostly holds up** — clinc 0.972, ag_news 0.821, massive
   0.811, banking77 0.742 (all within ~3–8 points of the 4.1-mini scores) — but collapses
   on trec (0.562 vs 0.881): the evolved trec prompts lean on fine-grained boundary rules
   the weaker model can't execute. Artifacts are model-portable in form, not always in
   performance.
6. **Two tasks are saturation-limited** and dilute all averages: clinc150 (~0.98
   everywhere) and emotion (~0.56 everywhere, where no arm beats the seed prompt
   meaningfully — consistent with the pre-library pilot on the same split). They are kept
   in the matrix for honesty, but they measure the ceiling, not the optimizer. Parse
   robustness held: ≤0.2 unparseable outputs per 300 test examples in every arm.

## Cost

Total measured spend for the entire matrix, including transfer evals: **$13.94**
(estimate upper bound was $29.41; temp-0 disk caching accounts for the difference).
A single full-layer run at light budget averaged **$0.21** optimize cost.

## What we'd change next (v0.2 candidates, in priority order)

1. Rework confusion-driven reflection: use the confusion matrix to *schedule* which
   boundary to fix but keep the reflective-example evidence primary; widen the component
   selector so non-confused labels still get periodic updates on large taxonomies.
2. Make the mining gate strict (`>` instead of `>=`, or require a margin) and screen
   candidate exemplars for label-noise (e.g., drop examples the taxonomy's own definitions
   contradict).
3. Headroom pre-screening in the harness: flag tasks where the seed prompt is within noise
   of the best arm (clinc150, emotion here) before spending the full matrix on them.
4. A properly-wired MIPROv2 arm for the paper.
