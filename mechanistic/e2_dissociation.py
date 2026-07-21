"""E2 — dissociation (C2): lessons are absorbed but inert.

Arms per eval task (banking77, trec, massive, bbh_geometric, gsm8k), per family:
  (a) vanilla GEPA            — test scores reused from E1 B1 cells (identical
                                config); one extra TRACED 1-seed run per task
                                provides the uptake false-positive baseline
  (b) lessons in reflection   — traced default-template proposer + lesson block
  (c) lessons at inference    — seed prompt + lessons appended, eval only
  (d) lessons in the INITIAL instruction, then vanilla optimization

Lesson banks are re-distilled per family from traced vanilla runs on THREE
SOURCE tasks disjoint from all eval tasks (ag_news, emotion, clinc150), using
the pilot's distillation prompt design.

UPTAKE METRIC (pre-specified before any arm-b run): fraction of reflection
responses in which an LLM judge (reflection model, fixed rubric below, temp 0)
finds >=1 bank lesson clearly applied in the diagnosis/proposal. The judge is
validated on 20 sampled cases hand-checked by the experimenter, agreement
reported. False-positive baseline: same judge on arm-(a) traces (no lessons
present).

Usage:
  python mechanistic/e2_dissociation.py --sources --family f1
  python mechanistic/e2_dissociation.py --distill --family f1
  python mechanistic/e2_dissociation.py --arm b --task trec --seed 40 --family f1
  python mechanistic/e2_dissociation.py --stream <name>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
sys.path.insert(0, str(HERE))
from e1_frontloading import (  # noqa: E402
    FAMILIES,
    GSM8K_SEED_PROMPT,
    QAAdapter,
    solve_batch,
)
from gepa.strategies.instruction_proposal import InstructionProposalSignature  # noqa: E402
from prep_data import load_task  # noqa: E402

import melvil as mv  # noqa: E402
from melvil.adapter import ClassificationAdapter  # noqa: E402
from melvil.artifact import BLOB_COMPONENT, render_prompt  # noqa: E402
from melvil.costs import lm_usage, usage_cost_usd  # noqa: E402
from melvil.lmutil import make_lm, reflection_callable  # noqa: E402
from melvil.optimize import seed_candidate_for  # noqa: E402
from melvil.program import classify_batch  # noqa: E402

RESULTS_DIR = HERE / "results_e2"
RUNS_DIR = HERE / "runs_e2"
LESSONS_DIR = HERE / "lessons_e2"
SOURCE_TASKS = ["ag_news", "emotion", "clinc150"]
EVAL_TASKS = ["banking77", "trec", "massive", "bbh_geometric", "gsm8k"]
SEEDS = [40, 41, 42, 43, 44]
BUDGET = 800

LESSON_BLOCK_HEADER = (
    "You also have access to a bank of lessons distilled from prompt-optimization runs on "
    "OTHER tasks. Apply the ones relevant to the failures you observe:\n"
)

DISTILL_PROMPT = """You are analyzing reflection traces from GEPA, a prompt-optimization algorithm, run on {n} different tasks: {tasks}. Each record shows the optimizer's diagnosis of failures and its proposed replacement instruction, with whether the proposal was accepted.

Distill 12-20 TASK-AGNOSTIC lessons about how to improve task prompts. Rules: each lesson generalizable (no task names, label names, or dataset specifics in the lesson text); ground each in the traces via an evidence field; derive from successes AND failures; deduplicate aggressively.

Return ONLY a JSON array: [{{"lesson": "<1-2 imperative sentences>", "type": "strategy"|"error-pattern", "evidence": [{{"task": "...", "what_happened": "..."}}]}}]

Trace records:
{records}
"""

UPTAKE_RUBRIC = """You are auditing whether a prompt-optimizer's reflection APPLIED any lesson from a fixed lesson bank. "Applied" means the diagnosis or the proposed instruction concretely uses the lesson's advice (same idea, any wording); merely being compatible with it does not count.

LESSON BANK:
{lessons}

REFLECTION (diagnosis + proposed instruction):
```
{reflection}
```

Return ONLY a JSON object: {{"applied": [<lesson numbers clearly applied>], "rationale": "<one sentence>"}}"""


class TracedProposer:
    """gepa default single-component proposal template, verbatim, with optional
    lesson block; logs every reflection to JSONL (pilot design, ported)."""

    def __init__(self, trace_path: Path, lessons: list[dict] | None, refl_call):
        self.trace_path = trace_path
        self.refl_call = refl_call
        if lessons:
            block = LESSON_BLOCK_HEADER + "\n".join(
                f"- [{le.get('type', 'strategy')}] {le['lesson']}" for le in lessons)
            self.template = InstructionProposalSignature.default_prompt_template.replace(
                "Provide the new instructions within ``` blocks.",
                block + "\nProvide the new instructions within ``` blocks.")
        else:
            self.template = InstructionProposalSignature.default_prompt_template

    def __call__(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for name in components_to_update:
            prompt = InstructionProposalSignature.prompt_renderer({
                "current_instruction_doc": candidate[name],
                "dataset_with_feedback": reflective_dataset[name],
                "prompt_template": self.template,
            })
            resp = self.refl_call(prompt)
            new = InstructionProposalSignature.output_extractor(resp)["new_instruction"]
            with open(self.trace_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "component": name,
                                    "response": resp, "proposed": new}) + "\n")
            out[name] = new
        return out


def _lm_pair(family: str, seed: int, qa: bool = False):
    fam = FAMILIES[family]
    task_lm = make_lm(fam["task"], 0.0, 400 if qa else 40)
    refl_lm = make_lm(fam["reflection"], 1.0, 8000, rollout_id=seed)
    return task_lm, refl_lm


def _gepa_run(task_name, family, seed, lessons, seed_instruction_extra, run_dir):
    """One traced GEPA run (classification or gsm8k QA), blob mode."""
    import gepa

    task = load_task(task_name)
    qa = task_name == "gsm8k"
    task_lm, refl_lm = _lm_pair(family, seed, qa)
    refl_call = reflection_callable(refl_lm)
    t0, r0 = len(task_lm.history), len(refl_lm.history)
    run_dir.mkdir(parents=True, exist_ok=True)
    proposer = TracedProposer(run_dir / "traces.jsonl", lessons, refl_call)

    if qa:
        seed_prompt = GSM8K_SEED_PROMPT + (seed_instruction_extra or "")
        adapter = QAAdapter(task_lm, task["dev"])
        adapter.propose_new_texts = proposer
        seed_cand = {"instruction": seed_prompt}
    else:
        spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
        seed_cand = seed_candidate_for(spec, mv.Features.none())
        seed_cand[BLOB_COMPONENT] += (seed_instruction_extra or "")
        cfg = mv.Config(task_model=FAMILIES[family]["task"],
                        reflection_model=FAMILIES[family]["reflection"],
                        budget=BUDGET, seed=seed, features=mv.Features.none())
        adapter = ClassificationAdapter(spec, task_lm, cfg, valset=task["dev"])
        adapter.proposer = proposer  # exposes via propose_new_texts property

    result = gepa.optimize(
        seed_candidate=seed_cand, trainset=task["train"], valset=task["dev"],
        adapter=adapter, reflection_lm=refl_call, max_metric_calls=BUDGET,
        run_dir=str(run_dir), seed=seed, display_progress_bar=False,
    )
    scores = list(result.val_aggregate_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    comp = "instruction" if qa else BLOB_COMPONENT
    best_prompt = result.candidates[best_idx][comp]
    if qa:
        test_preds = solve_batch(task_lm, best_prompt, task["test"])
        test_acc = sum(p["score"] for p in test_preds) / len(test_preds)
    else:
        preds = classify_batch(task_lm, render_prompt({comp: best_prompt}, task["labels"]),
                               task["test"], task["labels"], 8)
        test_acc = sum(1 for p in preds if p.correct) / len(preds)
    return {
        "test_accuracy": round(test_acc, 4),
        "best_dev": round(scores[best_idx], 4), "seed_dev": round(scores[0], 4),
        "curve": [{"metric_calls": int(result.discovery_eval_counts[i]),
                   "dev": round(scores[i], 4)} for i in range(len(scores))],
        "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0))
                          + usage_cost_usd(lm_usage(refl_lm, r0)), 4),
    }


def run_arm(arm: str, task_name: str, seed: int, family: str) -> dict:
    out_path = RESULTS_DIR / f"{arm}_{task_name}_{family}_s{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path.name}")
        return json.loads(out_path.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)
    lessons = None
    extra = None
    if arm in ("b", "c", "d"):
        lessons = json.loads((LESSONS_DIR / f"bank_{family}.json").read_text())[:10]
    if arm == "d":
        extra = "\n\nGuidance from experience on similar tasks:\n" + "\n".join(
            f"- {le['lesson']}" for le in lessons)
        lessons = None  # d = lessons at seed time only, vanilla reflection
    if arm == "c":  # inference-only: no optimization
        task = load_task(task_name)
        qa = task_name == "gsm8k"
        task_lm, _ = _lm_pair(family, seed, qa)
        t0 = len(task_lm.history)
        guidance = "\n\nGuidance from experience on similar tasks:\n" + "\n".join(
            f"- {le['lesson']}" for le in lessons)
        if qa:
            prompt = GSM8K_SEED_PROMPT + guidance
            preds = solve_batch(task_lm, prompt, task["test"])
            acc = sum(p["score"] for p in preds) / len(preds)
        else:
            spec = mv.TaskSpec.from_examples(task_name, task["train"] + task["dev"])
            cand = seed_candidate_for(spec, mv.Features.none())
            cand[BLOB_COMPONENT] += guidance
            cpreds = classify_batch(task_lm, render_prompt(cand, spec.label_names),
                                    task["test"], spec.label_names, 8)
            acc = sum(1 for p in cpreds if p.correct) / len(cpreds)
        rec = {"test_accuracy": round(acc, 4),
               "cost_usd": round(usage_cost_usd(lm_usage(task_lm, t0)), 4)}
    else:
        rec = _gepa_run(task_name, family, seed, lessons, extra,
                        RUNS_DIR / f"{arm}_{task_name}_{family}_s{seed}")
    rec.update(arm=arm, task=task_name, family=family, seed=seed)
    out_path.write_text(json.dumps(rec, indent=1))
    print(f"[done] {out_path.name}: {rec['test_accuracy']:.3f} (${rec['cost_usd']:.2f})")
    return rec


def run_sources(family: str) -> None:
    for t in SOURCE_TASKS:
        run_arm("a", t, 40, family)  # traced vanilla, one seed per source task


def distill(family: str) -> None:
    records = []
    for t in SOURCE_TASKS:
        tr = RUNS_DIR / f"a_{t}_{family}_s40" / "traces.jsonl"
        if not tr.exists():
            raise FileNotFoundError(f"run sources first: {tr}")
        for line in tr.read_text().splitlines():
            r = json.loads(line)
            records.append({"task": t, "reflection": r["response"][:2200]})
    _, refl_lm = _lm_pair(family, 0)
    out = reflection_callable(refl_lm)(DISTILL_PROMPT.format(
        n=len(SOURCE_TASKS), tasks=", ".join(SOURCE_TASKS),
        records=json.dumps(records, indent=0)))
    import re

    m = re.search(r"\[.*\]", out, re.DOTALL)
    bank = json.loads(m.group(0))
    LESSONS_DIR.mkdir(exist_ok=True)
    (LESSONS_DIR / f"bank_{family}.json").write_text(json.dumps(bank, indent=1))
    (LESSONS_DIR / f"distill_prompt_{family}.txt").write_text(DISTILL_PROMPT)
    print(f"{family}: distilled {len(bank)} lessons")


def judge_uptake(family: str, arms=("a", "b")) -> None:
    """Judge every reflection in traced runs of the given arms; write per-run
    uptake + a 20-case validation sample for hand-checking."""
    lessons = json.loads((LESSONS_DIR / f"bank_{family}.json").read_text())[:10]
    lesson_text = "\n".join(f"{i + 1}. {le['lesson']}" for i, le in enumerate(lessons))
    _, refl_lm = _lm_pair(family, 0)
    refl_lm.kwargs["temperature"] = 0.0
    call = reflection_callable(refl_lm)
    import re

    sample = []
    out = {}
    for run_dir in sorted(RUNS_DIR.glob(f"[ab]_*_{family}_s*")):
        arm = run_dir.name.split("_", 1)[0]
        if arm not in arms:
            continue
        tr = run_dir / "traces.jsonl"
        if not tr.exists():
            continue
        applied = 0
        records = [json.loads(x) for x in tr.read_text().splitlines()]
        for rec in records:
            resp = call(UPTAKE_RUBRIC.format(lessons=lesson_text,
                                             reflection=rec["response"][:3000]))
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            hits = json.loads(m.group(0)).get("applied", []) if m else []
            applied += bool(hits)
            if len(sample) < 20:
                sample.append({"run": run_dir.name, "reflection": rec["response"][:1500],
                               "judge": hits})
        out[run_dir.name] = {"n_reflections": len(records),
                             "uptake": round(applied / max(1, len(records)), 3)}
    (RESULTS_DIR / f"uptake_{family}.json").write_text(json.dumps(out, indent=1))
    (RESULTS_DIR / f"uptake_validation_sample_{family}.json").write_text(
        json.dumps(sample, indent=1))
    print(json.dumps(out, indent=1))


STREAMS = {
    f"e2_{fam}_{grp}": [(arm, t, s, fam) for t in tasks for arm in ("b", "d") for s in SEEDS]
    + [("c", t, 0, fam) for t in tasks] + [("a", t, 40, fam) for t in tasks]
    for fam in ("f1", "f2")
    for grp, tasks in [("x", ["banking77", "trec", "massive"]),
                       ("y", ["bbh_geometric", "gsm8k"])]
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", action="store_true")
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--family", default="f1", choices=("f1", "f2"))
    ap.add_argument("--arm", choices=("a", "b", "c", "d"))
    ap.add_argument("--task", choices=EVAL_TASKS + SOURCE_TASKS)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--stream", choices=list(STREAMS))
    args = ap.parse_args()
    if args.sources:
        run_sources(args.family)
    elif args.distill:
        distill(args.family)
    elif args.judge:
        judge_uptake(args.family)
    elif args.stream:
        for arm, t, s, fam in STREAMS[args.stream]:
            run_arm(arm, t, s, fam)
    elif args.arm and args.task:
        run_arm(args.arm, args.task, args.seed, args.family)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
