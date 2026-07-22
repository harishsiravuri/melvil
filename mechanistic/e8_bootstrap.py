"""E8 paired-bootstrap CIs + full per-task appendix table.

Recovers per-example correctness for the frozen E8 arms (seed / draft x2 / GEPA
reference) by re-evaluating each SAVED prompt over the same test set with the
temp-0 disk cache (the frozen pass evaluated these exact prompts on these exact
test sets, so this is ~100% cache hits and ~$0 — the script aborts if it finds
otherwise).

Statistic: for each (task, family), seed-average per-example correctness for
each arm over the identical test set, then paired bootstrap over test examples
(B=10000) for (i) the draft-x2 vs GEPA accuracy delta and (ii) the recovery
fraction (x2 - seed)/(GEPA - seed). Recovery is reported only where the GEPA
gain over seed exceeds the paired noise floor (else the task is saturated).

Prompt sources (matching what the frozen E8 pass used as references):
- seed / x1 / x2:  mechanistic/results_e8/frozen_{task}_{fam}_s{seed}.json['prompts']
- GEPA F1:         benchmarks/runs/{task}/vanilla-s{0,1,2} (v0.1) +
                   <snapshot>/runs_confirm/{task}/vanilla-s{10,11,12}
- GEPA F2:         mechanistic/runs_e8_gepa_f2/{task}/s{20..24}/artifact.json

Run: python mechanistic/e8_bootstrap.py
Outputs: mechanistic/results_sim/e8_bootstrap.json + prints the appendix table.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
from prep_data import load_task  # noqa: E402

from melvil.artifact import PromptArtifact  # noqa: E402
from melvil.costs import lm_usage  # noqa: E402
from melvil.lmutil import make_lm  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

RESULTS_E8 = HERE / "results_e8"
SNAPSHOT = Path("/tmp/melvil_snap/labelsmith/benchmarks/runs_confirm")
BENCH_RUNS = HERE.parent / "benchmarks" / "runs"
GEPA_F2 = HERE / "runs_e8_gepa_f2"
GEPA_F1 = HERE / "runs_e8_gepa_f1"
TASKS = ["banking77", "ag_news", "emotion", "trec", "clinc150", "massive",
         "stance_abortion", "sst5"]
FAMILIES = {"f1": "openrouter/openai/gpt-4.1-mini",
            "f2": "openrouter/anthropic/claude-haiku-4.5"}
DRAFT_SEEDS = [20, 21, 22, 23, 24]
B = 10000
LIVE_CALL_ABORT = 400  # a cache miss on this scale means a prompt-string mismatch


def gepa_prompts(task: str, family: str) -> list[str]:
    """Rendered best prompts for the E8 GEPA reference of (task, family)."""
    out = []
    if family == "f2":
        for s in DRAFT_SEEDS:
            p = GEPA_F2 / task / f"s{s}" / "artifact.json"
            if p.exists():
                out.append(PromptArtifact.load(p).render())
    else:
        for s in (0, 1, 2):  # v0.1 vanilla pool
            p = BENCH_RUNS / task / f"vanilla-s{s}" / "artifact.json"
            if p.exists():
                out.append(PromptArtifact.load(p).render())
        for s in (10, 11, 12):  # confirmation vanilla pool (snapshot)
            p = SNAPSHOT / task / f"vanilla-s{s}" / "artifact.json"
            if p.exists():
                out.append(PromptArtifact.load(p).render())
    return out


def gepa_prompts_seedmatched(task: str, family: str) -> list[tuple[int, str]]:
    """(seed, rendered prompt) for the seed-matched GEPA runs (seeds 20-24),
    enabling a true per-seed paired comparison against the draft arm."""
    base = GEPA_F1 if family == "f1" else GEPA_F2
    out = []
    for sd in DRAFT_SEEDS:
        p = base / task / f"s{sd}" / "artifact.json"
        if p.exists():
            out.append((sd, PromptArtifact.load(p).render()))
    return out


def paired_seed_diff(draft_by_seed, gepa_by_seed, rng):
    """Per-example paired diff averaged over matched seeds, then bootstrap over
    examples. draft_by_seed/gepa_by_seed: {seed: correctness-vector}."""
    seeds = sorted(set(draft_by_seed) & set(gepa_by_seed))
    n = len(next(iter(draft_by_seed.values())))
    per_ex = [statistics.mean(draft_by_seed[s][i] - gepa_by_seed[s][i] for s in seeds)
              for i in range(n)]
    point = statistics.mean(per_ex)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(per_ex[i] for i in idx) / n)
    boots.sort()
    return point, boots[int(0.025 * B)], boots[int(0.975 * B)], len(seeds)


def correctness(lm, prompt: str, test, labels, n_threads=8) -> list[int]:
    preds = classify_batch(lm, prompt, test, labels, n_threads)
    return [1 if p.correct else 0 for p in preds]


def seed_avg(vectors: list[list[int]]) -> list[float]:
    n = len(vectors[0])
    return [statistics.mean(v[i] for v in vectors) for i in range(n)]


def paired_bootstrap(a: list[float], b: list[float], rng: random.Random):
    """Bootstrap mean(a)-mean(b) over paired (same-index) test examples."""
    n = len(a)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def recovery_bootstrap(x2: list[float], seed: list[float], gepa: list[float],
                       rng: random.Random):
    n = len(x2)
    recs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d_x2 = sum(x2[i] - seed[i] for i in idx) / n
        d_g = sum(gepa[i] - seed[i] for i in idx) / n
        if abs(d_g) > 1e-6:
            recs.append(d_x2 / d_g)
    recs.sort()
    if len(recs) < 100:
        return None, None
    return recs[int(0.025 * len(recs))], recs[int(0.975 * len(recs))]


def main() -> None:
    out: dict = {"B": B, "tasks": {}, "note":
                 "paired bootstrap over identical test examples; seed-averaged "
                 "per-example correctness; recovered from temp-0 cache"}
    rng = random.Random(0)
    for family, task_model in FAMILIES.items():
        lm = make_lm(task_model, temperature=0.0, max_tokens=40)
        for task in TASKS:
            data = load_task(task)
            test, labels = data["test"], data["labels"]
            frozen = [json.loads((RESULTS_E8 / f"frozen_{task}_{family}_s{s}.json").read_text())
                      for s in DRAFT_SEEDS
                      if (RESULTS_E8 / f"frozen_{task}_{family}_s{s}.json").exists()]
            if not frozen:
                continue
            since = len(lm.history)
            seed_vec = correctness(lm, frozen[0]["prompts"]["seed"], test, labels)
            x2_by_seed = {f["seed"]: correctness(lm, f["prompts"]["x2"], test, labels)
                          for f in frozen}
            x2_vecs = list(x2_by_seed.values())
            gepa_vecs = [correctness(lm, pr, test, labels) for pr in gepa_prompts(task, family)]
            gepa_sm = {sd: correctness(lm, pr, test, labels)
                       for sd, pr in gepa_prompts_seedmatched(task, family)}
            u = lm_usage(lm, since)
            if u["calls"] > LIVE_CALL_ABORT:
                raise RuntimeError(
                    f"{task}/{family}: {u['calls']} live task-LM calls (expected ~0 "
                    f"cache hits). Prompt/test mismatch — aborting before spend.")
            if not gepa_vecs:
                continue

            seed_a = seed_vec  # deterministic prompt, one vector
            x2_a = seed_avg(x2_vecs)
            gepa_a = seed_avg(gepa_vecs)
            acc = lambda v: round(statistics.mean(v), 4)  # noqa: E731
            delta_lo, delta_hi = paired_bootstrap(x2_a, gepa_a, rng)
            gepa_gain = statistics.mean(gepa_a) - statistics.mean(seed_a)
            rec_point = ((statistics.mean(x2_a) - statistics.mean(seed_a)) / gepa_gain
                         if abs(gepa_gain) > 0.02 else None)
            rec_lo, rec_hi = (recovery_bootstrap(x2_a, seed_a, gepa_a, rng)
                              if rec_point is not None else (None, None))
            sm = None
            if gepa_sm:
                pt, lo, hi, nseed = paired_seed_diff(x2_by_seed, gepa_sm, rng)
                sm = {"delta": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "significant": not (lo <= 0 <= hi), "n_matched_seeds": nseed,
                      "gepa_acc": round(statistics.mean(seed_avg(list(gepa_sm.values()))), 4)}
            key = f"{task}/{family}"
            out["tasks"][key] = {
                "task": task, "family": family,
                "n_test": len(test), "n_draft_seeds": len(x2_vecs),
                "n_gepa_runs": len(gepa_vecs),
                "seed_acc": acc(seed_a), "x2_acc": acc(x2_a), "gepa_acc": acc(gepa_a),
                "delta_x2_minus_gepa": round(statistics.mean(x2_a) - statistics.mean(gepa_a), 4),
                "delta_ci95": [round(delta_lo, 4), round(delta_hi, 4)],
                "delta_significant": not (delta_lo <= 0 <= delta_hi),
                "recovery": round(rec_point, 3) if rec_point is not None else None,
                "recovery_ci95": ([round(rec_lo, 3), round(rec_hi, 3)]
                                  if rec_lo is not None else None),
                "saturated": rec_point is None,
                "cache_hits": u["cache_hits"], "live_calls": u["calls"],
                "seed_matched": sm,
                "gepa_reference": ("seed-pool (0-2,10-12)" if family == "f1"
                                   else "seed-matched (20-24)"),
            }
            r = out["tasks"][key]
            print(f"{key:<24} seed {r['seed_acc']:.3f} x2 {r['x2_acc']:.3f} "
                  f"GEPA {r['gepa_acc']:.3f} | dx2-GEPA {r['delta_x2_minus_gepa']:+.3f} "
                  f"[{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]"
                  f"{'*' if r['delta_significant'] else ''} | "
                  f"recov {r['recovery'] if r['recovery'] is not None else 'sat'}"
                  f"{r['recovery_ci95'] if r['recovery_ci95'] else ''} "
                  f"| sm dx2 {r['seed_matched']['delta']:+.3f}"
                  f"[{r['seed_matched']['ci95'][0]:+.3f},{r['seed_matched']['ci95'][1]:+.3f}]"
                  f"{'*' if r['seed_matched']['significant'] else ''}"
                  if r.get('seed_matched') else f"| live={r['live_calls']}")

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "e8_bootstrap.json").write_text(json.dumps(out, indent=1))

    # family-mean recovery with a bootstrap-of-task-means summary
    for fam in ("f1", "f2"):
        recs = [v["recovery"] for v in out["tasks"].values()
                if v["family"] == fam and v["recovery"] is not None]
        if recs:
            print(f"\n{fam}: mean recovery over non-saturated tasks {statistics.mean(recs):.2f} "
                  f"(n={len(recs)}); tasks where x2>GEPA with CI separation: "
                  f"{sum(1 for v in out['tasks'].values() if v['family']==fam and v['delta_significant'] and v['delta_x2_minus_gepa']>0)}")
    print("\nwrote results_sim/e8_bootstrap.json")


if __name__ == "__main__":
    main()
