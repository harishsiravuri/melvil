# Benchmark results — labelsmith

Three clearly separated evidence tiers, in chronological order:

1. **v0.1 exploration** (post-hoc analysis; seeds 0–2) — kept for the record.
2. **v0.2 iteration round** (targeted fixes + re-runs; seeds 0–2; exploratory).
3. **CONFIRMATION PASS** (pre-registered, frozen config, fresh seeds 10–12, two untouched
   datasets) — **the headline numbers. Nothing was changed after seeing them.**

**Headline (confirmation): at light budgets with a strong task model, vanilla GEPA over
labelsmith's deployable rendered prompt is the strongest arm.** It beats both MIPROv2 and
our own full classification layer:

| arm | banking77 | ag_news | emotion | trec | clinc150 | massive | stance_abortion | sst5 | mean |
|---|---|---|---|---|---|---|---|---|---|
| seed prompt | 0.760 | 0.830 | 0.563 | 0.600 | 0.980 | 0.840 | 0.500 | 0.547 | **0.703** |
| **vanilla GEPA** | 0.821 | 0.867 | 0.562 | 0.871 | 0.980 | 0.879 | 0.650 | 0.619 | **0.781** |
| labelsmith full (v0.2) | 0.821 | 0.843 | 0.556 | 0.848 | 0.978 | 0.837 | 0.563 | 0.596 | **0.755** |
| MIPROv2 (dspy-native) | 0.763 | 0.863 | 0.540 | 0.860 | 0.982 | 0.867 | 0.635 | 0.587 | **0.762** |

(Test accuracy, mean of 3 fresh seeds; macro-F1 table, per-seed values, and plot in
[confirmation_tables.md](confirmation_tables.md). Spend: $15.61.)

![confirmation](plots/confirmation_test_accuracy.png)

Setup for all tiers: task model `gpt-4.1-mini`, reflection/prompt model `gpt-4.1`
(OpenRouter); budget preset "light" per optimized arm; splits train 150 / dev 100 /
test ≤300 stratified (data seed 0); dev drives all optimization; test touched exactly once
per (task, arm, seed). MIPROv2 runs `auto="light"` in its native dspy ChatAdapter runtime
(its output is a dspy program, not a portable prompt string; measured task-model calls
~545–800 per run, comparable to our 800 cap).

---

## Confirmation pass — protocol and reading

**Protocol** (pre-registered in [run_confirmation.py](run_confirmation.py), committed before
any run): v0.2 code frozen after the iteration round; 8 datasets = the six iteration
datasets + two untouched ones (tweet_eval/stance_abortion, SetFit/sst5; trec-fine was
considered and rejected because its texts overlap trec-coarse); arms seed / vanilla / full /
MIPROv2; fresh seeds 10, 11, 12; test-only reporting.

**Reading the table, honestly:**

1. **The v0.1 result did not replicate.** In v0.1 exploration the full layer led vanilla by
   +1.2 mean points; under fresh seeds, new datasets, and leak-free mining it trails by
   −2.6. The v0.1 edge decomposes into (a) mining's leaky `>=` dev gate (fixed in v0.2 —
   after which exemplars were kept in **0/24** confirmation runs), (b) seed-level variance
   (per-seed spreads up to ±0.05 on the same arm/task), and (c) task selection — v0.1's
   headroom tasks had no hard small-taxonomy task.
2. **Why the codebook loses at light budgets: update coverage, not generalization.** On
   stance_abortion, full's *dev* accuracy only reached 0.590 vs vanilla's 0.793 — the
   structured prompt failed to optimize, with 2 of 3 seeds stuck near the seed prompt.
   A light budget accepts ~6–10 proposals; vanilla revises the entire prompt with each one,
   while the codebook spends each proposal on one or two components. Whole-prompt rewrites
   can discover global strategies (e.g. how to treat neutral reporting in stance detection)
   that per-definition edits cannot assemble in so few updates. The one task where full
   matched vanilla (banking77, 25 labels, 0.821 = 0.821, and the best macro-F1 of any arm at
   0.813) is consistent with this: with many labels, per-label structure is closer to the
   right granularity — but at this budget it never got *ahead*.
3. **Vanilla GEPA + rendered-prompt runtime > MIPROv2** (0.781 vs 0.762 mean) — with the
   format caveat noted above. The deployable-prompt GEPA loop is a strong, simple baseline,
   and it is the configuration we now recommend as labelsmith's default story at light
   budgets (`Features.none()`).
4. **Parse robustness held everywhere** (≤1 unparseable output per 300 across all arms —
   the fixed output contract does its job in both prompt styles).

## v0.2 iteration round (exploratory; seeds 0–2)

Changes made after (and because of) v0.1, before the confirmation freeze:

- **Confusion selector fixed**: deficit-weighted round-robin over the top-5 confused pairs
  replaced greedy worst-pair selection (v0.1 had sent 36/42 updates to 2 of 4 labels).
  Re-running full vs no_confusion on the three headroom tasks: banking77 +2.1,
  massive +1.0, ag_news −1.4 → mean **+0.6 for confusion-reflection, up from −1.1** under
  the v0.1 selector. Directionally vindicated (large taxonomies), still not decisive.
- **Mining hardened**: label-noise consistency prescreen + optional LLM screen against the
  taxonomy's own definitions + quarantined accept gate (exemplars excluded from the gating
  eval; must beat base by more than the binomial noise floor). Result: **18/18 iteration
  runs and 24/24 confirmation runs rejected all exemplars** — every v0.1 "kept" decision
  had been within the noise floor. Hard-example mining at light budgets, honestly measured,
  is a no-op; it remains in the library as a safety-gated feature.
- `ls.screen()` added (headroom verdict before spending budget) — a user-facing product of
  these experiments: clinc150 and emotion would have been flagged `saturated`/ceilinged
  before costing anything.

## v0.1 exploration (superseded; kept for the record)

Mean test accuracy over 6 datasets (seeds 0–2): seed 0.762, vanilla 0.817, full 0.829,
ablations attributing the margin to the codebook. Superseded by the confirmation pass; the
detailed v0.1 tables and plots remain in [results_tables.md](results_tables.md) and
`results/`. Total v0.1 matrix spend $13.94.

## What we take away (paper framing)

- **Positive result**: GEPA-style reflective evolution over a *deployable rendered prompt*
  (not a framework-native program) is a strong classification optimizer: +7.8 points over
  the seed prompt on average, ahead of MIPROv2, with a portable prompt artifact as output.
- **Negative result, cleanly established**: our classification-specific layer (per-label
  codebook + confusion-driven reflection + hard-example mining) does not pay for itself at
  light budgets on 3–30-label tasks, and the appearance that it did in exploration was a
  reproducibility lesson (gate leakage + seed variance). The confirmation-pass discipline
  caught it before publication.
- **Open question worth one more study**: budget scaling. The coverage argument (point 2
  above) predicts the codebook should catch up and pass vanilla as accepted-proposal counts
  grow, and banking77's macro-F1 hints the crossover is nearest on large taxonomies. That
  is a hypothesis for medium/heavy budgets, not a claim.

## Spend

v0.1 matrix $13.94 · iteration round $2.04 · MIPROv2 validation $0.18 · confirmation
$15.61 · **total benchmarks ≈ $31.8** (plus ~$0.35 smoke tests).
