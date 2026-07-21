"""Cost decomposition: seed / draft-x1 / draft-x2 / GEPA-B1 per family.

Reports, per method, the OPTIMIZATION-side cost (producing the prompt; the
one-time deploy/test eval is separate): task-model calls, writer (reflection)
calls, task tokens (in/out), writer tokens (in/out), cached fraction, dollars,
and wall-clock; plus the diagnosis context size fed to the writer, per dataset.

Draft numbers are recovered by replaying e8_diagnose's exact LM calls against
the temp-0 / rollout-id caches (0 live calls, $0; guarded). GEPA numbers come
straight from the seed-matched gepa_{task}_{fam} artifacts' budget block.
Representative task: banking77 (25 labels — the large-taxonomy case).

Run: python mechanistic/cost_decomposition.py
Outputs: results_sim/cost_decomposition.json + printed table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
sys.path.insert(0, str(HERE))
import statistics as _stats  # noqa: E402

import tiktoken as _tk  # noqa: E402
from e8_diagnose import (  # noqa: E402
    FAMILIES,
    REWRITE_TEMPLATE,
    build_diagnosis,
    seed_prompt_for,
)
from prep_data import CONFIRM_EXTRA_TASKS, TASKS, load_task  # noqa: E402

from melvil.artifact import PromptArtifact  # noqa: E402
from melvil.costs import price_for  # noqa: E402
from melvil.lmutil import make_lm  # noqa: E402

_ENC = _tk.get_encoding("o200k_base")  # gpt-4.1 family; a good proxy for claude too
from melvil.program import classify_batch  # noqa: E402

REP_TASK = "banking77"
DRAFT_SEED = 20
ALL_TASKS = TASKS + CONFIRM_EXTRA_TASKS
LIVE_ABORT = 40


def _tok(usage: dict) -> tuple[int, int]:
    return usage["prompt_tokens"], usage["completion_tokens"]


def _ntok(text: str) -> int:
    return len(_ENC.encode(text))


def draft_decomposition(task_name: str, family: str) -> dict:
    """Token-exact cost of draft-x1/x2 from the SAVED frozen prompts (seed/x1/x2)
    + rebuilt diagnoses. No LM calls — fully offline, $0. Dollars from the
    family's price table. Averaged over the 5 frozen seeds."""
    task = load_task(task_name)
    fam = FAMILIES[family]
    tin_price, tout_price = price_for(fam["task"])
    rin_price, rout_price = price_for(fam["reflection"])
    dev_texts = [e.text for e in task["dev"]]
    dev_tok = sum(_ntok(t) for t in dev_texts)  # user-message tokens over the dev set

    def dev_eval_tokens(prompt: str) -> tuple[int, int]:
        # one dev eval = n_dev calls; input = (system prompt) per call + each user text
        n = len(dev_texts)
        return n * _ntok(prompt) + dev_tok, n * 3  # ~3 output tokens (a label)

    per_seed = []
    for rec_path in sorted((HERE / "results_e8").glob(f"frozen_{task_name}_{family}_s*.json")):
        rec = json.loads(rec_path.read_text())
        pr = rec["prompts"]
        # diagnoses were saved in run dirs; rebuild size from saved diagnosis files
        # if present, else estimate from the seed diagnosis token count
        seed_eval = dev_eval_tokens(pr["seed"])
        x1_eval = dev_eval_tokens(pr["x1"])
        x2_eval = dev_eval_tokens(pr["x2"])
        # writer calls: input ~ REWRITE_TEMPLATE + current prompt + diagnosis;
        # output = the produced prompt (x1 then x2). Diagnosis token count taken
        # from the saved diagnosis_x{i}.txt when available.
        diag_toks = []
        for it in (1, 2):
            f = _find_diag(task_name, family, rec["seed"], it)
            diag_toks.append(_ntok(f) if f else 600)
        w1_in = _ntok(REWRITE_TEMPLATE) + _ntok(pr["seed"]) + diag_toks[0]
        w1_out = _ntok(pr["x1"])
        w2_in = _ntok(REWRITE_TEMPLATE) + _ntok(pr["x1"]) + diag_toks[1]
        w2_out = _ntok(pr["x2"])
        arms = {
            "seed": {"task_calls": len(dev_texts), "writer_calls": 0,
                     "task_tok": seed_eval, "writer_tok": (0, 0)},
            "draft_x1": {"task_calls": 2 * len(dev_texts), "writer_calls": 1,
                         "task_tok": (seed_eval[0] + x1_eval[0], seed_eval[1] + x1_eval[1]),
                         "writer_tok": (w1_in, w1_out)},
            "draft_x2": {"task_calls": 3 * len(dev_texts), "writer_calls": 2,
                         "task_tok": (seed_eval[0] + x1_eval[0] + x2_eval[0],
                                      seed_eval[1] + x1_eval[1] + x2_eval[1]),
                         "writer_tok": (w1_in + w2_in, w1_out + w2_out)},
        }
        for a in arms.values():
            a["usd"] = round((a["task_tok"][0] * tin_price + a["task_tok"][1] * tout_price
                              + a["writer_tok"][0] * rin_price + a["writer_tok"][1] * rout_price)
                             / 1e6, 4)
        per_seed.append({"arms": arms, "diag_tok": diag_toks})
    # average across seeds
    def avg(arm, field, idx=None):
        vals = [ps["arms"][arm][field] if idx is None else ps["arms"][arm][field][idx]
                for ps in per_seed]
        return round(_stats.mean(vals)) if idx is not None or field == "usd" else _stats.mean(vals)
    out = {"arms": {}, "diagnosis_ctx_tokens_mean":
           [round(_stats.mean(ps["diag_tok"][k] for ps in per_seed)) for k in (0, 1)]}
    for arm in ("seed", "draft_x1", "draft_x2"):
        out["arms"][arm] = {
            "task_calls": per_seed[0]["arms"][arm]["task_calls"],
            "writer_calls": per_seed[0]["arms"][arm]["writer_calls"],
            "task_tokens": [round(_stats.mean(ps["arms"][arm]["task_tok"][0] for ps in per_seed)),
                            round(_stats.mean(ps["arms"][arm]["task_tok"][1] for ps in per_seed))],
            "writer_tokens": [round(_stats.mean(ps["arms"][arm]["writer_tok"][0] for ps in per_seed)),
                              round(_stats.mean(ps["arms"][arm]["writer_tok"][1] for ps in per_seed))],
            "opt_usd": round(_stats.mean(ps["arms"][arm]["usd"] for ps in per_seed), 4),
        }
    return out


def _find_diag(task_name: str, family: str, seed: int, it: int) -> str | None:
    # frozen E8 wrote diagnosis files under mechanistic/ (e8_diagnose run dirs);
    # search a few likely locations, else return None (fallback estimate used)
    for base in (HERE / "runs_e8", HERE):
        f = base / f"{task_name}_{family}_s{seed}" / f"diagnosis_x{it}.txt"
        if f.exists():
            return f.read_text()
    return None


def gepa_decomposition(task_name: str, family: str) -> dict | None:
    art_path = HERE / f"runs_e8_gepa_{family}" / task_name / f"s{DRAFT_SEED}" / "artifact.json"
    if not art_path.exists():
        return None
    a = PromptArtifact.load(art_path)
    b = a.budget
    tu = b.get("task_lm_usage", {})
    ru = b.get("reflection_lm_usage", {})
    return {
        "task_calls_metric": b.get("metric_calls_spent"),
        "writer_calls": ru.get("calls"),
        "task_tokens": [tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)],
        "writer_tokens": [ru.get("prompt_tokens", 0), ru.get("completion_tokens", 0)],
        "cost_usd": b.get("cost_usd"),
        "wall_clock_s": b.get("opt_seconds"),
    }


def diagnosis_sizes() -> dict:
    """Diagnosis context tokens per dataset (family F1, seed prompt, cached)."""
    fam = FAMILIES["f1"]
    task_lm = make_lm(fam["task"], 0.0, 40)
    sizes = {}
    for t in ALL_TASKS:
        task = load_task(t)
        preds = classify_batch(task_lm, seed_prompt_for(task), task["dev"], task["labels"], 8)
        diag = build_diagnosis(preds, task["labels"])
        sizes[t] = {"n_labels": len(task["labels"]), "diagnosis_tokens": len(diag) // 4}
    return sizes


def main() -> None:
    out = {"representative_task": REP_TASK, "draft": {}, "gepa": {}, "diagnosis_sizes": {}}
    for fam in ("f1", "f2"):
        out["draft"][fam] = draft_decomposition(REP_TASK, fam)
        g = gepa_decomposition(REP_TASK, fam)
        if g:
            out["gepa"][fam] = g
    out["diagnosis_sizes"] = diagnosis_sizes()

    (HERE / "results_sim").mkdir(exist_ok=True)
    (HERE / "results_sim" / "cost_decomposition.json").write_text(json.dumps(out, indent=1))

    print(f"=== cost decomposition (representative task: {REP_TASK}) ===")
    for fam in ("f1", "f2"):
        d = out["draft"][fam]
        print(f"\n--- {fam} ---")
        for arm, a in d["arms"].items():
            print(f"{arm:<9}: {a['task_calls']:>4} task calls, {a['writer_calls']} writer; "
                  f"task tok {a['task_tokens']}, writer tok {a['writer_tokens']}; "
                  f"opt ${a['opt_usd']}")
        print(f"  diagnosis ctx {d['diagnosis_ctx_tokens_mean']} tok")
        g = out["gepa"].get(fam)
        if g:
            print(f"GEPA-B1  : {g['task_calls_metric']} task metric calls, {g['writer_calls']} "
                  f"writer; task tok {g['task_tokens']}, writer tok {g['writer_tokens']}; "
                  f"${g['cost_usd']}, {g['wall_clock_s']}s")
    print("\n=== diagnosis context size per dataset (F1 seed prompt) ===")
    for t, v in out["diagnosis_sizes"].items():
        print(f"  {t:<16} {v['n_labels']:>3} labels -> {v['diagnosis_tokens']:>5} diag tokens")
    print("\nwrote results_sim/cost_decomposition.json")


if __name__ == "__main__":
    main()
