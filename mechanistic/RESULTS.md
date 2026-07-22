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

> **Cost note (corrected post-freeze, see Paper-support §4):** the "~10–15% of GEPA" figure is loose. Token-exact, draft-×2 *optimization* is ~40% (F1) / ~23% (F2) of GEPA-B1; the ~1/10 figure holds only for the single-rewrite ×1 variant on F2. The frozen accuracy numbers above are unchanged.

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
roughly a quarter-to-third of its optimization cost (a tenth for the single
rewrite variant on cheaper reflection) — supporting a screen → diagnose-then-write →
evolve-if-headroom pipeline — but on tasks requiring globally restructured
prompts, iterative evolution remains necessary."* Iteration-pass numbers held
under fresh seeds (banking77 0.849→0.843, trec 0.856→0.852). **Significance (paired bootstrap, added post-freeze; does NOT alter the frozen
point estimates).** Per-example correctness was recovered for every arm from
the temp-0 cache (0 live calls, $0; `e8_bootstrap.py`) and bootstrapped over
the identical test set (B=10,000; full per-task table in the appendix below).
The point-estimate "beats GEPA on 2 of 8 tasks" softens under CIs to **one
CI-separated win — banking77 — in BOTH families** (Δ +0.029 [+0.011, +0.048]
on F1; +0.034 [+0.011, +0.057] on F2). Draft-×2 is a **statistical tie** with
GEPA on 4 tasks per family (the saturated/near-ceiling ones), and CI-separated
*below* GEPA on the genuinely hard tasks (F1: ag_news, stance_abortion, sst5;
F2: ag_news, trec, stance_abortion). This tightens the verdict rather than
overturning it: draft matches full evolution wherever the task is easy or
mid-difficulty, wins outright on the one large-taxonomy intent task, and is
beaten only where global prompt restructuring matters — precisely the
escalation boundary the library encodes (draft → screen → optimize). (The loss/tie split above uses the frozen unpaired pool; the seed-matched re-analysis in Paper-support §1–2 is the version to cite — it moves ag_news to a tie and trec to a loss, headline win on banking77 unchanged.)

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

## E1 — front-loading (C1) · **COMPLETE** ($93.74)

Full matrix: GEPA × {800, 1600, 3200} × 5 tasks (banking77/trec/massive +
bbh_geometric + gsm8k) × 5 fresh seeds (30–34) on F1, with per-proposal-attempt
instrumentation; GEPA-F2 (claude) B1/B2 × 3 seeds; MIPROv2-F1 by preset tier
(light/medium/heavy; GSM8K-MIPROv2 dropped — budget parity uninterpretable;
TextGrad skipped — no maintained harness within a day). 165 runs, zero failures.

**Headline: C1 DIED as stated.** Median fraction of total dev gain from the
FIRST accepted proposal: **0.33** on F1 (n=73), **0.42** on F2 (n=30) — flat
across budgets B1→B3 (no concentration even at 4× budget), and strongly
task-dependent (trec 0.62–0.64, massive 0.29–0.33, banking77 0.00, gsm8k
0.00–0.25). Half of the gain needs ~2 accepts (median by-2 = 0.50); the rest
accrues across later accepts. MIPROv2's apparent 0.71 is measured at its own
full-eval checkpoints (2–4 per run) — a granularity artifact of its evaluation
schedule, reported with that caveat, so the cross-optimizer contrast is
qualified rather than claimed.

![cumulative gain](plots/e1_cumulative_gain.png)

Paper sentence: *"Reflective prompt evolution's gains are distributed, not
front-loaded: across budgets, tasks, and two model families, the first accepted
rewrite carries a median one-third of the final improvement — refuting the
folk model in which one good rewrite does the work, and explaining why methods
that bank on early signal (racing, aggressive stopping, best-of-K round one)
underdeliver."* Secondary finding for practitioners: budget scaling is strongly
diminishing — 4× budget buys +1.4 test points on average (GEPA F1 0.811 →
0.825); GEPA beats MIPROv2 at every budget tier (replicating the benchmark
confirmation).

E1's 105 GEPA curves were folded into the simulation pool; the E5-A frontier
re-computed on 222 informative runs is unchanged in shape (A(2): 63% retention
at 38% budget; A(3): 75% at 51%; G(200): 90% at 92%) — C5's verdict stands
with better power.

## E7 — best-of-K first rewrite (C7) · **COMPLETE** ($6.37)

K=4 diverse first rewrites (distinct reflection cache namespaces + distinct
error samples), 25-example dev screen, winner promoted, vanilla continuation at
matched total budget; banking77/trec/massive × seeds 30–34 vs the E1 vanilla
B1 cells (identical seeds — paired comparison).

**Verdict: C7 DIED (clean null).** Paired mean test delta **+0.002 ± 0.009**
(banking77 +0.001, trec +0.010, massive −0.005). The screen itself works — the
K rewrites spread by a mean 7.2 accuracy points on the screen set and the
winner's first full-dev score averages 0.822 — but the head start washes out:
with gains distributed across many accepts (E1), evolution re-converges
regardless of the starting rewrite. Paper sentence: *"Investing budget in a
better first rewrite buys nothing at matched total budget (+0.2 ± 0.9 points):
the optimizer's later accepts redistribute whatever the first rewrite missed."*
Together with E6 (racing dead) and E5 (modest stopping frontier), all three
early-signal exploits fail for the same measured reason — a coherent negative
triad grounded in the E1 result.

## E4 — coarse-to-fine (C4) · iteration complete; **no confirmation warranted** ($4.58)

Iteration (marked exploratory; seeds 30–32, B1, 4 tasks): vanilla / full-layer /
c2f(rejections:2) / c2f(fraction:0.6, banking77 ablation).

| task | vanilla | full | c2f(rej2) | c2f(frac60) |
|---|---|---|---|---|
| banking77 | 0.826 | 0.821 | 0.818 | 0.786 |
| trec | 0.892 | 0.873 | 0.892 | — |
| massive | 0.858 | 0.850 | 0.858 | — |
| stance_abortion | 0.640 | 0.537 | 0.607 | — |

**Verdict: C4 DIED in iteration; freezing and confirming it would confirm a
non-method.** Mechanism, precisely: (i) the stall trigger (2 consecutive
rejections) fires so late that 14/15 runs had no budget left for phase 2 —
GEPA keeps accepting sporadically to the end (another face of the distributed-
gains result E1); c2f then equals vanilla minus overhead. (ii) When phase 2 is
forced early (fraction 0.6), refinement actively subtracts (−4.0 on banking77):
decomposed per-label editing degrades a good blob rather than sharpening it.
Paper sentence: *"Sequencing whole-prompt evolution before per-label refinement
does not rescue structure: evolution rarely stalls early enough to fund a
refinement phase, and when refinement is forced, it hurts."* The decomposition
machinery (LLM split + verification gate) worked as designed — the gate
correctly caught degraded decompositions — so the negative is attributable to
the refinement phase itself, not implementation failure.

## E5 Phase B — LIVE adaptive stopping (C5) · **COMPLETE** ($0.61 marginal)

Rule from the enriched Phase-A frontier (stop after 3 accepts OR 150-call
patience), run live on banking77/trec/massive × seeds 30–34, PAIRED against
the E1 vanilla B1 cells (same seeds/config, full budget).

**Verdict: C5 DIED as an accuracy-preserving claim.** Stopped runs spend
**48.9%** of the metric calls and lose **2.8 ± 1.5 test points** (banking77
−4.1, trec −3.3, massive −0.9). The live test refutes the rescue hypothesis
that late accepts only chase dev noise — they carry real test value, as the
distributed-gains result (E1) predicts. (Measured dollar cost of the stopped
runs was ~12% of baseline, but that is a disk-cache artifact of re-running
prefixes of already-cached trajectories; metric calls are the honest unit.)
Paper sentence: *"Early stopping is a dial, not a free lunch: halving the
budget costs ~3 test points, because reflective evolution's later accepts are
not noise."* C5 survives only as that explicit trade-off table row.

## E3 — structure, extraction family (C3) · **COMPLETE** ($13.78)

MIT Restaurant slot filling (tner parquet; text-in/JSON-out; micro slot-F1 on
type-sensitive normalized-span multisets), GEPA B1, free-text vs per-field
codebook, both families, seeds 30–34 (paired).

| family | free-text | per-field codebook | paired Δ |
|---|---|---|---|
| F1 | 0.636 ± 0.018 | 0.674 ± 0.030 | +0.037 ± 0.039 |
| F2 | 0.685 ± 0.029 | 0.727 ± 0.029 | **+0.042 ± 0.019** |

**⚠ SUPERSEDED IN PART by the extraction factorial (Paper-support §5): the extraction advantage is mostly field *definitions*, not fielded *structure*; the shape-match claim does not survive the 2×2. Read that section with this one.**

**Verdict: C3 RESOLVED as shape-match — the campaign's first positive
structure result.** Per-field structure wins on extraction (+4 slot-F1 points,
CI-separated on F2, same direction on F1) while per-label structure loses on
classification (benchmark confirmation + E4). The classification budget cells
(from E1) complete the other half of C3: structure does not recover at 2–4×
budget either (vanilla itself gains only +1.4 at 4×). Paper sentence:
*"Prompt structure pays when it mirrors the task's output structure — per-field
prompts beat free text on slot filling in both model families — and costs when
it merely mirrors the label space, as in classification."* Caveat, stated: the
codebook arm's seed includes per-field definitions the free seed lacks (seed
dev 0.65 vs 0.62), so the +4 bundles structured seeding with structured
evolution; we treat the bundle as the method, and note the free arm had the
full budget to absorb definitions and did not close the gap.

## Library alignment (v0.4)

As of melvil v0.4.0 the library leads with the same method the campaign
confirmed: `mv.draft()` (the E8 procedure, ported verbatim from
`e8_diagnose.py`) is the primary entry point and the README's first code
example; full GEPA evolution is the documented escalation path
(draft → screen → optimize). Library and paper now make the same claim with
the same evidence.

## E2 — dissociation (C2) · **COMPLETE** ($99.83 + $3 judging)

Full rerun of the pilot's design at campaign scale: per-family lesson banks
re-distilled from traced vanilla runs on three disjoint source tasks (F1: 16
lessons, F2: 12); arms (a) vanilla [test scores reused from E1 B1 cells],
(b) lessons in reflection, (c) lessons at inference only, (d) lessons in the
initial instruction then vanilla optimization; 5 eval tasks including
non-classification (bbh_geometric, gsm8k) × both families × 5 seeds.

Mean test-accuracy deltas vs vanilla: **(b) −1.4 (F1) / +0.8 (F2)** points —
within noise, both signs; **(d) +0.4 / −0.3** — null; **(c) −11 / −17** —
inference-only lessons lose badly everywhere (and catastrophically break
claude on BBH: 0.09, a formatting failure worth a footnote).

**Uptake metric: the pre-specified judge FAILED ITS CONTROL, and that is the
finding.** Arm-(a) reflections (which never saw a lesson) are judged to
"apply" lessons at the same rate as arm-(b) (F1: 1.00 vs 1.00; F2: 0.71 vs
0.69; validation sample inspected — e.g. a no-lesson ag_news reflection judged
to apply 6 lessons). The distilled lessons are generic good practice that
reflection already performs unprompted; "absorption" is unmeasurable because
there is nothing distinctive to absorb.

**Verdict: C2 SURVIVED, sharpened.** The dissociation replicates at scale and
beyond classification — lessons help nowhere they're injected (reflection or
seed) and hurt at inference — and the mechanism is now cleaner than
"absorbed but inert": *distilled cross-task lessons are redundant with what
reflective optimization already does*, which is why injecting them changes
nothing and why the pilot's vocabulary-overlap "uptake" was real but causally
empty. Paper sentence: *"Cross-task experience distilled into lessons is
redundant, not inert: an uptake judge with a no-lesson control shows vanilla
reflections already exhibit every behavior the lessons prescribe, explaining
why injection changes nothing (Δ ≤ 1.4 points, both families) while the same
text at inference costs 11–17 points."*

## Paper-support computations (added post-campaign; free/cache-backed except items 2 & 5)

Five follow-ups requested for the paper's central table and appendices. Two of them **corrected earlier claims** — flagged inline. Total new spend $14.5 (seed-matched GPT GEPA $11.0 + extraction factorial $3.4; all bootstrap/cost/sensitivity work was cache-backed at $0).

### 1–2. E8 CIs with seed-matched GEPA (closes the unpaired gap)

The frozen F1 GEPA reference reused the v0.1/confirmation vanilla pool (seeds 0–2, 10–12) — not seed-matched to the draft arm (seeds 20–24). We ran fresh F1 vanilla-GEPA at seeds 20–24 on all 8 tasks ($11.0) so both families now have a **true per-seed paired** bootstrap (draft seed *s* vs GEPA seed *s* on the identical test set, B=10,000). Δ = draft-×2 − GEPA, `*` = 95% CI excludes 0.

| task | F1 seed-matched Δ | F1 (unpaired pool) Δ | F2 seed-matched Δ |
|---|---|---|---|
| banking77 | +0.031 [+0.009,+0.054]* | +0.029* | +0.034 [+0.011,+0.057]* |
| ag_news | -0.008 [-0.024,+0.008] | -0.025* | -0.027 [-0.051,-0.005]* |
| emotion | -0.009 [-0.030,+0.011] | -0.007 | +0.024 [-0.001,+0.051] |
| trec | -0.025 [-0.045,-0.005]* | -0.015 | -0.029 [-0.051,-0.008]* |
| clinc150 | +0.005 [-0.003,+0.015] | +0.006 | +0.002 [-0.003,+0.009] |
| massive | +0.005 [-0.015,+0.024] | +0.003 | -0.006 [-0.029,+0.017] |
| stance_abortion | -0.163 [-0.209,-0.116]* | -0.177* | -0.022 [-0.036,-0.008]* |
| sst5 | -0.036 [-0.067,-0.005]* | -0.050* | -0.005 [-0.037,+0.027] |

F1 seed-matched: WIN ['banking77']; tie ['ag_news', 'emotion', 'clinc150', 'massive']; loss ['trec', 'stance_abortion', 'sst5'].

F2 seed-matched: WIN ['banking77']; tie ['emotion', 'clinc150', 'massive', 'sst5']; loss ['ag_news', 'trec', 'stance_abortion'].

**Headline unchanged** (one CI-separated win per family — banking77). Seed-matching shifts two F1 cells honestly vs the pool: ag_news goes loss→**tie** (−0.008, ns) and trec goes tie→**loss** (−0.025*); the pool had slightly mis-estimated both. The seed-matched numbers are the ones to print.

### 3. First-accept-share sensitivity + early-to-final rank correlation

First-accept gain share (GEPA F1, all budgets/tasks) under three definitions — the front-loading-is-dead result is robust:

| definition | median | mean ± 95% CI | n |
|---|---|---|---|
| V1 running-max share | 0.333 | 0.331 ± 0.065 | 73 |
| V2 absolute gain (acc. pts) | 0.020 | 0.068 ± 0.021 | 73 |
| V3 first-*improver* share | 0.417 | 0.491 ± 0.057 | 73 |

By task type (V1): bbh-reasoning 0.46 (n=21); classification 0.33 (n=63); math-qa 0.33 (n=19)
 · exclusions: 2/105 runs flat (total gain <0.005).

Even the most generous variant (V3, counting only value-improving accepts) puts the first improver at a median 0.42 — still far from a majority. Early-to-final dev **rank correlation** (Spearman across seeds, racing pool), the mechanism behind E6's death:

| checkpoint F | mean Spearman | n_tasks |
|---|---|---|
| 0.15 | +0.419 ± 0.276 | 8 |
| 0.25 | +0.314 ± 0.258 | 9 |
| 0.40 | +0.535 ± 0.255 | 9 |

Early standing is a weak-to-moderate predictor (ρ ≈ 0.3–0.5, CIs nearly spanning 0) — positive but far too noisy for early selection to pay for parallel starts at matched budget.

### 4. Cost decomposition — **corrects the "~1/10" framing**

Token-exact optimization cost (producing the prompt; deploy/test eval excluded), representative task banking77, from saved prompts via tiktoken ($0):

| method | task calls | writer calls | task tok (in/out) | writer tok (in/out) | opt $ |
|---|---|---|---|---|---|
| seed (f1) | 100 | 0 | 16,030/300 | 0/0 | $0.007 |
| draft_x1 (f1) | 200 | 1 | 105,980/600 | 911/886 | $0.052 |
| draft_x2 (f1) | 300 | 2 | 222,690/900 | 2,561/2,040 | $0.112 |
| **GEPA-B1 (f1)** | 890 | 11 | 509,226/3,792 | 6,379/6,761 | **$0.277** (235s) |
| seed (f2) | 100 | 0 | 16,030/300 | 0/0 | $0.018 |
| draft_x1 (f2) | 200 | 1 | 59,420/600 | 911/421 | $0.071 |
| draft_x2 (f2) | 300 | 2 | 128,270/900 | 2,096/1,096 | $0.155 |
| **GEPA-B1 (f2)** | 851 | 7 | 543,413/6,181 | 5,222/4,914 | **$0.664** (183s) |

**CORRECTION.** The earlier "~1/10 cost" / "~10–15% of GEPA" framing was loose. Token-exact, draft-×2 optimization is **40% (F1) / 23% (F2)** of GEPA-B1 — closer to a quarter–third, not a tenth. The ~1/10 figure only holds for the single-rewrite ×1 variant on the cheaper-reflection family (×1 = 19% F1 / 11% F2). Draft's cost is dominated by re-evaluating the dev set (iterations+1) times; GEPA's by its ~9× more task calls and ~5× more writer calls. The README/docstring "~1/10" wording should be corrected to "~1/4–1/3 (×2), ~1/10 (×1)" in a v0.4.1.

Diagnosis context size scales with label count, not linearly: banking77 589tok/25lab, ag_news 528tok/4lab, emotion 465tok/6lab, trec 313tok/6lab, … (313–589 tokens across the 8 datasets).

### 5. Extraction 2×2 factorial — **demotes the E3 shape-match claim**

The E3 verdict credited a +4 slot-F1 extraction win to per-field *structure*, with a caveat that the codebook arm also carried field *definitions* the free arm lacked. The 2×2 factorial (structure: blob|fielded × definitions: absent|present; F1, 5 seeds, $3.4) resolves it:

| | definitions absent | definitions present |
|---|---|---|
| **blob** | 0.636 ± 0.018 | 0.673 ± 0.018 |
| **fielded** | 0.659 ± 0.048 | 0.674 ± 0.030 |

Main effects (paired by seed, n=5):
- **structure** (fielded − blob): **+0.012 ± 0.032** — not CI-separated
- **definitions** (present − absent): **+0.026 ± 0.034** — larger, marginal
- interaction: -0.022 ± 0.070

**CORRECTION.** The extraction advantage is carried more by the field **definitions** (+0.026) than by fielded **structure** (+0.012), and at n=5 neither main effect is individually CI-separated from zero. The original E3 free-vs-codebook +0.038 bundled both factors; decomposed, ~2/3 is semantic content (definitions), ~1/3 is layout (structure). **The shape-match hypothesis is NOT confirmed by the factorial** — 'structure wins on extraction' overstated it. The honest claim: *adding field definitions helps extraction modestly; fielded structure per se does not measurably help at this power.* This weakens C3's positive half; the classification half (structure loses) stands.

## Campaign closeout

**No further confirmation pass is needed, by design audit:** E8 already ran
under a frozen pre-registered protocol with fresh seeds (its numbers are the
headline); E7/E5-B used paired designs on fresh seeds against matched E1
cells; E3-extraction has 5 paired seeds with CIs; E4/E6 died in
iteration/simulation (nothing to confirm); E1/E2 are measurement studies, not
tunable methods. Remaining balance and calendar go to the paper.

Campaign spend: E8 $42.62 · E1 $93.74 · E2 $102.8 · E3-extraction $13.78 ·
E4 $4.58 · E7 $6.37 · E5-B $0.61 · simulations $0 · misc ≈ $1 →
**≈ $266 total** (brief envelope $220–400). Zero failed runs across ~560
optimization/evaluation runs; every result checkpointed and committed.

## The paper's figures (generated, in plots/)

1. `e1_cumulative_gain.png` — THE mechanistic figure: cumulative gain by
   accepted-proposal index, per task × budget; median first-accept share 0.33.
2. `e5_frontier.png` — stopping-rule cost-retention frontier (222 runs): the
   no-free-lunch trade, with the E5-B live −2.8-point check in the caption.
3. `e6_frontier.png` — racing ≤ 0 at matched budget in every cell.
4. E8 recovery table (from `results_sim/e8_summary.json`) rendered as the
   headline bar figure for draft-vs-GEPA per task × family (to typeset).
5. E3-extraction vs classification structure contrast (two-panel; to typeset
   from `results_e3x/` + benchmark tables).

## Which method should the paper lead with — the memo

**Ranking by confirmed effect size × cost × story:**

1. **E8 / `draft()` — LEAD.** Only surviving positive method. Frozen numbers:
   60–91% of GEPA's gain at ~1/4–1/3 the optimization cost (~1/10 for the
   single-rewrite ×1 variant), seed-matched CI-separated win on banking77 in
   both families + statistical ties across the mid range (Paper-support §1–2,4).
   It is also now the library's primary API (v0.4.0), so the paper and the
   artifact tell one story. Frame: error-grounded direct writing, not
   evolution-acceleration (C1's death is what makes this framing correct).
2. **E1 distributed-gains — the SPINE.** Not a method but the paper's central
   measurement: median first-accept share 0.33 (robust across three definitions,
   Paper-support §3), flat across budgets/families. It explains every negative
   (E5/E6/E7 — the weak +0.3–0.5 early-to-final rank correlation is the
   mechanism) and motivates E8's framing. Lead section after the intro.
3. **E2 redundancy / E3 structure — the negative-nuance pair.** E2: lessons are
   redundant with default reflection (failed-control uptake finding — a methods
   contribution about LLM-judged "uptake"). E3: structure LOSES on classification
   and, per the 2×2 factorial (Paper-support §5), its apparent extraction win is
   mostly field *definitions*, not layout — the shape-match hypothesis did not
   survive. Together: "prompt structure rarely pays; when it seems to, check
   whether it is the content."
4. **E2 redundancy — the dissociation, sharpened.** Strong negative with a
   novel measurement twist (the failed-control uptake finding is itself a
   methods contribution about LLM-judged "uptake" metrics).
5. **E5 stopping — one honest table row** (the dial: −2.8 pts at 49% calls).
   E4/E6/E7 appear as the coherent negative triad, each one paragraph.

Suggested title framing: *"Distributed, Not Front-Loaded: What Reflective
Prompt Evolution Actually Buys, and When One Rewrite Is Enough."*

### Appendix — E8 per-task paired-bootstrap table

Per-example correctness recovered from the temp-0 cache (0 live calls, $0); seed-averaged over the frozen draft seeds (20–24) and the E8 GEPA reference pool; paired bootstrap over the identical test set, B=10,000. Δ = draft-×2 accuracy − GEPA accuracy; `*` = 95% CI excludes 0. Recovery = (×2−seed)/(GEPA−seed), shown only where GEPA's gain over seed clears the noise floor (else *sat* = saturated).


**F1 (gpt-4.1-mini / gpt-4.1)** — GEPA reference = v0.1+confirmation vanilla pool

| task | seed | draft ×2 | GEPA | Δ (×2−GEPA) | 95% CI | recovery | recovery 95% CI |
|---|---|---|---|---|---|---|---|
| banking77 | 0.760 | 0.843 | 0.813 | +0.029* | [+0.011, +0.048] | 1.55 | [1.18, 2.43] |
| ag_news | 0.830 | 0.838 | 0.863 | -0.025* | [-0.047, -0.004] | 0.24 | [-0.27, 0.74] |
| emotion | 0.563 | 0.547 | 0.553 | -0.007 | [-0.030, +0.016] | sat | — |
| trec | 0.600 | 0.852 | 0.867 | -0.015 | [-0.035, +0.005] | 0.94 | [0.88, 1.02] |
| clinc150 | 0.980 | 0.986 | 0.980 | +0.006 | [-0.003, +0.017] | sat | — |
| massive | 0.840 | 0.867 | 0.864 | +0.003 | [-0.015, +0.020] | 1.12 | [0.00, 2.66] |
| stance_abortion | 0.500 | 0.473 | 0.650 | -0.177* | [-0.241, -0.112] | -0.18 | [-0.70, 0.06] |
| sst5 | 0.547 | 0.569 | 0.619 | -0.050* | [-0.088, -0.014] | 0.30 | [-0.43, 0.69] |

**F2 (claude-haiku-4.5 / sonnet-4.5)** — GEPA reference = fresh vanilla GEPA seeds 20–24

| task | seed | draft ×2 | GEPA | Δ (×2−GEPA) | 95% CI | recovery | recovery 95% CI |
|---|---|---|---|---|---|---|---|
| banking77 | 0.767 | 0.810 | 0.776 | +0.034* | [+0.011, +0.057] | sat | — |
| ag_news | 0.867 | 0.853 | 0.880 | -0.027* | [-0.050, -0.005] | sat | — |
| emotion | 0.553 | 0.568 | 0.544 | +0.024 | [-0.003, +0.051] | sat | — |
| trec | 0.827 | 0.852 | 0.881 | -0.029* | [-0.051, -0.008] | 0.47 | [-0.21, 0.84] |
| clinc150 | 0.980 | 0.982 | 0.980 | +0.002 | [-0.003, +0.009] | sat | — |
| massive | 0.833 | 0.861 | 0.867 | -0.006 | [-0.029, +0.017] | 0.82 | [-0.23, 2.04] |
| stance_abortion | 0.529 | 0.666 | 0.689 | -0.022* | [-0.036, -0.008] | 0.86 | [0.71, 0.95] |
| sst5 | 0.580 | 0.575 | 0.579 | -0.005 | [-0.038, +0.028] | sat | — |

F1 significance: draft-×2 CI-separated WIN on ['banking77']; statistical tie on ['emotion', 'trec', 'clinc150', 'massive']; CI-separated loss on ['ag_news', 'stance_abortion', 'sst5'].

F2 significance: draft-×2 CI-separated WIN on ['banking77']; statistical tie on ['emotion', 'clinc150', 'massive', 'sst5']; CI-separated loss on ['ag_news', 'trec', 'stance_abortion'].
