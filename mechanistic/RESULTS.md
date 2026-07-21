# Mechanistic campaign — results

Claims C1–C8 from the campaign brief; one section per experiment, each with an
honest verdict (survived / narrowed / died) and the sentence we'd put in the
paper. Iteration vs confirmation outputs are marked. Baseline reference:
`benchmarks/RESULTS.md` confirmation pass (vanilla GEPA 0.781 mean test, fresh
seeds 10–12, budget 780/"light").

## E5 Phase A — adaptive-budget stopping rules (C5) · SIMULATION, $0 · **complete**

Setup: 156 logged GEPA runs pooled from the v0.1 matrix, v0.2 iteration, and
the confirmation pass (`curve_pool.jsonl`); rules computable exactly from
accepted-candidate curves — A(k) = stop after the k-th accepted candidate,
G(C) = stop after C metric calls without a new accept. (Proposal-count rules
like "k consecutive rejections" are NOT simulable from our logs: curves cannot
distinguish gepa's skip iterations from rejected proposals; noted for E1's
instrumentation, which will log both.) 119 runs with total gain ≥ 0.005 scored.

| rule | % budget spent | % dev gain retained | median | full-retention runs |
|---|---|---|---|---|
| A(1) | 30% ± 1% | 47.8% ± 7.3% | 50% | 24% |
| **A(2)** | **49% ± 2%** | **72.2% ± 6.3%** | **89%** | 44% |
| A(3) | 67% ± 3% | 85.1% ± 4.5% | 100% | 61% |
| G(120) | 46% ± 5% | 34.3% ± 8.0% | 0% | 24% |
| G(200) | 87% ± 4% | 83.0% ± 6.7% | 100% | 82% |
| G(300) | 98% ± 1% | 99.2% ± 1.7% | 100% | 99% |

![frontier](plots/e5_frontier.png)

**Verdict: C5 NARROWED.** The success bar (≥90–95% retention at ≤50% budget)
is not met: the best ~50%-budget rule (A(2)) retains 72% of dev gain (median
89% — the mean is dragged by a minority of runs whose late accepts matter).
Two structural reasons, both worth reporting: (i) full dev evals dominate
accepted-candidate cost (~106 calls each of 800), bounding what any stopping
rule can save; (ii) late accepts contribute more than the front-loading story
predicted — which also feeds the C1 verdict below. Paper sentence: *"Simple
stopping rules trade roughly linearly: ~70% of the achievable gain at half the
budget, ~85% at two-thirds — useful budget knobs, but no free lunch; the
retention ceiling is set by evaluation overhead, not proposal count."*
Phase B (live) decision: run only if E1's richer instrumentation suggests a
hybrid rule (accept-count + patience) clearing ~85% retention at ≤60%; the
pure rules here don't justify prospective spend yet.

Side-finding feeding C1: across 119 runs the FIRST accepted proposal carries a
median 50% (mean 48%) of total dev gain — substantial, but far from the "large
majority" the pilot's single-task observation suggested. C1 is already
weakening before E1 runs; E1 will quantify across budgets/optimizers/families.

## E6 Phase A — GEPA-Race simulation (C6) · SIMULATION, $0 · **complete**

Setup: race N ∈ {2,3,4,6} logged same-config vanilla trajectories (6 seeds/task
on six tasks, 3 on two), checkpoint F ∈ {15%,25%,40%}, winner = dev leader at
F·T. Two accountings, per the brief. 90–122 seed-subsets per cell (sampled).
Positioning: CAPO (arXiv:2504.16005) races candidate prompts within a run; this
races whole trajectories.

Matched TOTAL budget (winner truncated to T·(1−(N−1)F)): **every cell ≤ 0**
(best −0.0001, worst −0.0106 dev vs the mean single full run). Race-to-full
(1.3–2.5× budget): +0.1 to +0.4 test points vs the mean seed — within or near
CI of zero — and **always negative vs the luckiest seed** (−1.1 to −3.0).

![racing](plots/e6_frontier.png)

**Verdict: C6 DIED in simulation.** Early-checkpoint dev rank is too weakly
predictive of final rank for trajectory selection to pay for its parallel
starts; at honest matched-budget accounting racing never wins, and the
race-to-full gains are luck-harvesting that a "report the luckiest seed"
skeptic already owns. Paper sentence: *"Racing optimization trajectories
recovers only seed-selection luck: at matched total budget it never beats a
single full run, because early dev standing is a poor predictor of final
standing."* E6 Phase B: **cancelled** (simulation is decisive; the live
mechanism it can't capture — different post-checkpoint reflections — has no
channel to overcome a negative matched-budget baseline). This negative is
itself paper material alongside C1's weakening: both say the early signal is
noisier than the front-loading story implied.

## E8 — diagnose-then-write (C8) · **FROZEN RESULTS** (no changes after these numbers)

Frozen pass complete (8 tasks × 2 families × seeds 20–24 + GEPA-F2 references;
$42.62). Mean test accuracy across the 8 tasks:

| family | seed | ×1 | ×2 | GEPA B1 | ×2 recovery of GEPA gain | ×2 optimize cost |
|---|---|---|---|---|---|---|
| F1 gpt-4.1-mini/4.1 | 0.703 | 0.740 | 0.747 | 0.776 | **60%** | ~10–15% of GEPA's |
| F2 haiku-4.5/sonnet-4.5 | 0.742 | 0.761 | 0.771 | 0.774 | **91%** | ~10–15% of GEPA's |

Per-task recovery is strongly heterogeneous (F1: banking77 155%, massive 112%,
trec 94% — but stance_abortion **−18%**, ag_news 24%, sst5 30%; F2: stance 86%,
massive 82%, trec 47%, with banking77/emotion exceeding GEPA outright).
clinc150 flat/saturated both families, sst5 flat on F2. GEPA-F1 reference =
existing vanilla pool (seeds 0–2, 10–12); GEPA-F2 = fresh runs (seeds 20–24).

**Verdict: C8 NARROWED to a strong cost result, not a replacement claim.**
The ≥80–90%-recovery-at-≤15%-cost bar is met on F2 (91%) but not on F1's mean
(60%) — and F1's failures cluster exactly where whole-prompt iteration
discovers global strategies a one-shot diagnosis can't (stance_abortion, the
same task that broke the per-label codebook). Paper sentence: *"A single
error-grounded rewrite recovers 60–91% of GEPA's gain (family-dependent) at
about a tenth of its cost — supporting a screen → diagnose-then-write →
evolve-if-headroom pipeline — but on tasks requiring globally restructured
prompts, iterative evolution remains necessary."* Iteration-pass numbers held
under fresh seeds (banking77 0.849→0.843, trec 0.856→0.852). Pending for the
final tables: paired bootstrap CIs (per-example scores recomputable from saved
prompts via the temp-0 cache at ~zero cost).

### Iteration-phase record (pre-freeze)

Iteration (banking77 + trec, F1, seeds 0–1, ~$0.91): seed→×1→×2 test accuracy
0.760→0.811→**0.849** on banking77 (GEPA reference 0.821 — exceeded) and
0.600→0.792→**0.856** on trec (GEPA 0.871 — 94% of gain recovered), at
~15–25% of GEPA's optimization spend. Headline candidate.

**FROZEN protocol (registered before any frozen run):** code as committed at
freeze; all 8 prepped datasets; families F1 (gpt-4.1-mini / gpt-4.1) and F2
(claude-haiku-4.5 / claude-sonnet-4.5); fresh seeds 20–24; arms seed / ×1 / ×2
from one script run per (task, family, seed). GEPA references: F1 = existing
vanilla runs (seeds 0–2 v0.1 + 10–12 confirmation, reported as such); F2 = new
vanilla GEPA at B1, seeds 20–24. Metrics: test accuracy, macro-F1, recovery
fraction (arm − seed)/(GEPA − seed), cost per arm; mean ± 95% CI over seeds;
paired bootstrap vs GEPA on identical test sets. No changes after numbers.

## Pending experiments
- **E1 front-loading (C1)** — harness next; its runs also enrich this
  simulation pool.
- **E2 dissociation (C2), E3 structure (C3), E4 coarse-to-fine (C4),
  E7 best-of-K (C7), E5 Phase B** — queued per campaign order; E7's rationale
  is weakened (not killed) by the C1 side-finding: the first accept is ~50% of
  gain, so improving it still matters, but expectations are tempered.
