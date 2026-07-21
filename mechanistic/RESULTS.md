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

## Pending experiments

- **E8 diagnose-then-write (C8)** — in progress (iteration phase).
- **E1 front-loading (C1)** — harness next; its runs also enrich this
  simulation pool.
- **E2 dissociation (C2), E3 structure (C3), E4 coarse-to-fine (C4),
  E7 best-of-K (C7), E5 Phase B** — queued per campaign order; E7's rationale
  is weakened (not killed) by the C1 side-finding: the first accept is ~50% of
  gain, so improving it still matters, but expectations are tempered.
