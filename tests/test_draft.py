"""Offline tests for draft(): diagnosis assembly, artifact fields,
iterations=1 vs 2, cost accounting, screen(artifact), optimize(start_from)."""

import re

import pytest

from melvil import Config, Features, TaskSpec, draft, estimate_draft_cost, optimize, screen
from melvil.artifact import BLOB_COMPONENT, OUTPUT_CONTRACT
from melvil.data import Example
from melvil.draft import build_diagnosis
from melvil.program import Prediction
from melvil.testing import FakeReflectionLM, FakeTaskLM, patch_lms

LABELS = ["alpha", "beta", "gamma", "delta"]


def make_data():
    train = [Example(f"the {lab} signal in sample {i}", lab) for lab in LABELS for i in range(6)]
    dev = [Example(f"dev {lab} obs {i}", lab) for lab in LABELS for i in range(6)]
    return train, dev


class DraftFakeReflectionLM(FakeReflectionLM):
    """Understands the draft rewrite template: returns a full prompt with rich
    per-label definitions (which FakeTaskLM's quality metric rewards)."""

    def __call__(self, prompt=None, messages=None, **kw):
        text = str(prompt if prompt is not None else messages)
        if "expert prompt engineer" in text and "DIAGNOSIS" in text:
            self._record(len(text) // 4, 300)
            m = re.search(r"exactly as given: (.*?)\.\n", text)
            labels = [x.strip() for x in m.group(1).split(",")] if m else LABELS
            self.n += 1
            defs = "\n".join(
                f"- {lab}: texts clearly about {lab}, marked by {lab}-specific vocabulary "
                f"and context (revision {self.n})." for lab in labels)
            body = (f"Classify the text into exactly one category.\n\nCategories:\n{defs}\n\n"
                    f"Boundary rules:\nWhen two categories seem plausible, prefer the one "
                    f"whose specific vocabulary appears verbatim.\n\n{OUTPUT_CONTRACT}")
            return [f"Here is the improved prompt.\n```\n{body}\n```"]
        return super().__call__(prompt=prompt, messages=messages, **kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train, dev = make_data()
    spec = TaskSpec.from_examples("draft-task", train)
    gold = {e.text: e.label for e in [*train, *dev]}
    task_lm = FakeTaskLM(gold, spec.label_names)
    refl_lm = DraftFakeReflectionLM()
    patch_lms(monkeypatch, task_lm, refl_lm)
    return spec, train, dev, task_lm, refl_lm


def cfg(run_dir, **kw):
    return Config(task_model="fake/task", reflection_model="fake/reflection",
                  num_threads=4, run_dir=str(run_dir), **kw)


def test_build_diagnosis_contents():
    preds = [Prediction("t1", "alpha", "beta", "beta", False, "fb"),
             Prediction("t2", "alpha", "alpha", "alpha", True, "fb"),
             Prediction("t3", "beta", "", "??", False, "fb")]
    d = build_diagnosis(preds, ["alpha", "beta"])
    assert "recall" in d and "alpha" in d
    assert "misread as" in d
    assert "failed to parse" in d


def test_draft_improves_and_records(env, tmp_path):
    spec, train, dev, task_lm, refl_lm = env
    artifact = draft(spec, train, dev, cfg(tmp_path / "d1"))
    assert artifact.scores["dev_accuracy"] > artifact.scores["seed_dev_accuracy"]
    assert artifact.budget["method"] == "draft"
    assert artifact.budget["iterations"] == 2
    assert artifact.budget["cost_usd"] >= 0
    assert artifact.notes.startswith("draft x2")
    assert BLOB_COMPONENT in artifact.components
    assert OUTPUT_CONTRACT in artifact.render()
    assert len(artifact.scores["dev_accuracy_per_iteration"]) == 3
    assert len(artifact.config["draft_diagnoses"]) == 2
    assert (tmp_path / "d1" / "diagnosis_x1.txt").exists()
    assert (tmp_path / "d1" / "diagnosis_x2.txt").exists()
    assert (tmp_path / "d1" / "artifact.json").exists()


def test_draft_single_iteration_and_resume(env, tmp_path):
    spec, train, dev, *_ = env
    a1 = draft(spec, train, dev, cfg(tmp_path / "d2"), iterations=1)
    assert a1.budget["iterations"] == 1
    assert not (tmp_path / "d2" / "diagnosis_x2.txt").exists()
    a2 = draft(spec, train, dev, cfg(tmp_path / "d2"), resume=True)
    assert a2.artifact_id == a1.artifact_id
    with pytest.raises(FileExistsError):
        draft(spec, train, dev, cfg(tmp_path / "d2"))


def test_screen_accepts_artifact(env, tmp_path):
    spec, train, dev, *_ = env
    artifact = draft(spec, train, dev, cfg(tmp_path / "d3"))
    res_seed = screen(spec, dev, cfg(tmp_path / "s1"))
    res_draft = screen(artifact, dev, cfg(tmp_path / "s2"))
    assert res_draft.seed_accuracy >= res_seed.seed_accuracy  # drafted prompt is better
    assert res_draft.verdict in {"saturated", "marginal", "headroom"}


def test_optimize_start_from_draft(env, tmp_path):
    spec, train, dev, *_ = env
    drafted = draft(spec, train, dev, cfg(tmp_path / "d4"))
    art = optimize(spec, train, dev,
                   cfg(tmp_path / "o1", budget=260, features=Features.none()),
                   start_from=drafted)
    assert art.parent_id == drafted.artifact_id
    # optimization began from the drafted prompt, not the bare seed
    assert art.scores["seed_dev_accuracy"] == pytest.approx(
        drafted.scores["dev_accuracy"], abs=0.15)
    with pytest.raises(ValueError, match="blob mode"):
        optimize(spec, train, dev,
                 cfg(tmp_path / "o2", budget=260, features=Features.all()),
                 start_from=drafted)


def test_estimate_draft_cost(env, tmp_path):
    spec, train, dev, *_ = env
    c = Config(task_model="openai/gpt-4.1-mini", reflection_model="openai/gpt-4.1")
    e1 = estimate_draft_cost(spec, dev, c, iterations=1)
    e2 = estimate_draft_cost(spec, dev, c, iterations=2)
    assert 0 < e1.total_usd < e2.total_usd
    assert "ESTIMATED TOTAL" in str(e2)
