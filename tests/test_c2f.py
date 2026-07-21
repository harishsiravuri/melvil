"""Offline tests for coarse-to-fine (melvil.c2f): phase-1 stall switch,
LLM decomposition parsing, verification gate, and phase-2 refinement."""

import re

from melvil import Config, TaskSpec
from melvil.artifact import BLOB_COMPONENT, TASK_COMPONENT, label_component
from melvil.c2f import decompose_prompt, optimize_c2f
from melvil.data import Example
from melvil.testing import FakeReflectionLM, FakeTaskLM, patch_lms

LABELS = ["alpha", "beta", "gamma", "delta"]


def make_data():
    train = [Example(f"the {lab} signal in sample {i}", lab) for lab in LABELS for i in range(8)]
    dev = [Example(f"dev {lab} obs {i}", lab) for lab in LABELS for i in range(6)]
    return train, dev


class C2FReflectionLM(FakeReflectionLM):
    """Handles decompose prompts properly; otherwise defers to the stock fake
    (blob rewrites / tagged component blocks)."""

    def __call__(self, prompt=None, messages=None, **kw):
        text = str(prompt if prompt is not None else messages)
        if "Split the following classification system prompt" in text:
            self._record(len(text) // 4, 200)
            m = re.search(r"PROMPT:\n```\n(.*?)\n```", text, re.DOTALL)
            blob = m.group(1) if m else "instructions"
            blocks = [f"```component: task_instruction\n{blob[:200]}\n```"]
            blocks += [
                f"```component: label::{lab}\nTexts that clearly express {lab}; "
                f"the deciding feature is explicit {lab}-specific vocabulary. "
                "Decide by feature f1 when ambiguous. Decide by feature f2 when ambiguous.\n```"
                for lab in LABELS
            ]
            blocks.append("```component: boundary_rules\nPick the more specific category. "
                          "Decide by feature f3 when ambiguous.\n```")
            return ["\n".join(blocks)]
        return super().__call__(prompt=prompt, messages=messages, **kw)


def test_decompose_prompt_parses_and_maps_labels():
    fake = C2FReflectionLM()
    spec = TaskSpec.from_examples("t", make_data()[0])

    def call(p):
        return fake(p)[0]

    comps = decompose_prompt("Classify into alpha, beta, gamma, delta.", spec, call)
    assert TASK_COMPONENT in comps and "boundary_rules" in comps
    for lab in LABELS:
        assert label_component(lab) in comps
        assert "vocabulary" in comps[label_component(lab)]


def test_c2f_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train, dev = make_data()
    spec = TaskSpec.from_examples("c2f-task", train)
    gold = {e.text: e.label for e in [*train, *dev]}
    task_lm = FakeTaskLM(gold, spec.label_names)
    refl_lm = C2FReflectionLM()
    patch_lms(monkeypatch, task_lm, refl_lm)

    cfg = Config(task_model="fake/task", reflection_model="fake/reflection",
                 budget=420, num_threads=4, run_dir=str(tmp_path / "run"))
    artifact = optimize_c2f(spec, train, dev, cfg, switch=("rejections", 2))

    assert artifact.scores["dev_accuracy"] >= artifact.scores["seed_dev_accuracy"]
    assert "c2f switch=('rejections', 2)" in artifact.notes
    phases = {c.get("phase") for c in artifact.curve}
    assert 1 in phases  # phase 1 always present
    assert (tmp_path / "run" / "decomposition.json").exists()
    # final components are either a refined codebook or the phase-1 blob
    assert (TASK_COMPONENT in artifact.components) or (BLOB_COMPONENT in artifact.components)
    # resume returns the saved artifact
    again = optimize_c2f(spec, train, dev, cfg, switch=("rejections", 2), resume=True)
    assert again.artifact_id == artifact.artifact_id


def test_c2f_fraction_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train, dev = make_data()
    spec = TaskSpec.from_examples("c2f-frac", train)
    gold = {e.text: e.label for e in [*train, *dev]}
    patch_lms(monkeypatch, FakeTaskLM(gold, spec.label_names), C2FReflectionLM())
    cfg = Config(task_model="fake/task", reflection_model="fake/reflection",
                 budget=420, num_threads=4, run_dir=str(tmp_path / "runf"))
    artifact = optimize_c2f(spec, train, dev, cfg, switch=("fraction", 0.6))
    assert "fraction" in artifact.notes
