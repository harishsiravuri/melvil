"""Full offline end-to-end: optimize -> artifact -> evaluate -> report, with
fake LMs (no API keys, no network). The fake task LM's accuracy rises with
prompt quality, so the optimizer has a genuine gradient and its accept/reject
machinery is exercised for real."""

import pytest

from melvil import Config, Features, TaskSpec, evaluate, optimize, report
from melvil.artifact import BLOB_COMPONENT, TASK_COMPONENT, PromptArtifact, label_component
from melvil.data import Example
from melvil.testing import fake_lms_for, patch_lms

LABELS = ["alpha", "beta", "gamma", "delta"]


def make_data(n_per_train=8, n_per_dev=6):
    train = [
        Example(f"the {lab} signal appears in sample {i}", lab)
        for lab in LABELS
        for i in range(n_per_train)
    ]
    dev = [
        Example(f"dev {lab} observation {i}", lab) for lab in LABELS for i in range(n_per_dev)
    ]
    return train, dev


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # runs/ and any cache dirs land in tmp
    train, dev = make_data()
    spec = TaskSpec.from_examples("offline-task", train)
    task_lm, refl_lm = fake_lms_for(train, dev, spec.label_names, seed=0)
    patch_lms(monkeypatch, task_lm, refl_lm)
    return spec, train, dev, task_lm, refl_lm


def make_config(tmp_path=None, **kw):
    kw.setdefault("budget", 260)
    kw.setdefault("num_threads", 4)
    return Config(task_model="fake/task", reflection_model="fake/reflection", **kw)


def test_full_pipeline_all_features(env, tmp_path):
    spec, train, dev, task_lm, refl_lm = env
    cfg = make_config(run_dir=str(tmp_path / "run1"))
    rounds = []
    artifact = optimize(spec, train, dev, cfg, on_round=lambda info: rounds.append(info))

    # the optimizer actually improved on the (description-free) seed prompt
    assert artifact.scores["seed_dev_accuracy"] is not None
    assert artifact.scores["dev_accuracy"] > artifact.scores["seed_dev_accuracy"]
    # codebook structure survived
    assert TASK_COMPONENT in artifact.components
    assert all(label_component(lb) in artifact.components for lb in LABELS)
    # rounds fired with monotonically nondecreasing best score
    assert rounds and all(
        rounds[i].best_dev_score <= rounds[i + 1].best_dev_score for i in range(len(rounds) - 1)
    )
    # budget accounting present and sane
    assert artifact.budget["metric_calls_spent"] > 0
    assert artifact.budget["task_lm_usage"]["calls"] > 0
    # curve exists and is serialized
    assert artifact.curve and artifact.curve[0]["metric_calls"] == 0
    # run dir artifacts
    run_dir = tmp_path / "run1"
    assert (run_dir / "artifact.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "traces.jsonl").exists()

    # evaluate + report (fake LM again; transfer flag off)
    rep = evaluate(artifact, dev, model="fake/task", num_threads=4)
    assert rep.n == len(dev)
    assert 0.0 <= rep.accuracy <= 1.0 and 0.0 <= rep.macro_f1 <= 1.0
    md = report(rep)
    assert "| alpha |" in md and "accuracy" in md
    md2 = report(artifact)
    assert artifact.artifact_id in md2


def test_resume_returns_saved_artifact(env, tmp_path):
    spec, train, dev, *_ = env
    cfg = make_config(run_dir=str(tmp_path / "run2"))
    a1 = optimize(spec, train, dev, cfg)
    a2 = optimize(spec, train, dev, cfg, resume=True)
    assert a2.artifact_id == a1.artifact_id
    with pytest.raises(FileExistsError):
        optimize(spec, train, dev, cfg)


def test_vanilla_mode_is_single_blob(env, tmp_path):
    spec, train, dev, *_ = env
    cfg = make_config(run_dir=str(tmp_path / "run3"), features=Features.none())
    artifact = optimize(spec, train, dev, cfg)
    assert set(artifact.components) == {BLOB_COMPONENT}
    assert artifact.render().startswith("Classify the given text")


def test_ablation_confusion_off(env, tmp_path):
    spec, train, dev, *_ = env
    cfg = make_config(
        run_dir=str(tmp_path / "run4"),
        features=Features(codebook=True, confusion_reflection=False, hard_example_mining=False),
    )
    artifact = optimize(spec, train, dev, cfg)
    assert TASK_COMPONENT in artifact.components
    assert artifact.exemplars == []


def test_artifact_diff_across_runs(env, tmp_path):
    spec, train, dev, *_ = env
    a = optimize(spec, train, dev, make_config(run_dir=str(tmp_path / "r5")))
    b = optimize(
        spec, train, dev, make_config(run_dir=str(tmp_path / "r6"), seed=1), parent=a
    )
    assert b.parent_id == a.artifact_id
    d = b.diff(a)
    assert a.artifact_id in d and b.artifact_id in d


def test_validation_rejects_unknown_labels(env, tmp_path):
    spec, train, dev, *_ = env
    bad_dev = [*dev, Example("mystery", "unknown_label")]
    with pytest.raises(ValueError, match="unknown_label"):
        optimize(spec, train, bad_dev, make_config(run_dir=str(tmp_path / "r7")))


def test_loaded_artifact_renders_identically(env, tmp_path):
    spec, train, dev, *_ = env
    a = optimize(spec, train, dev, make_config(run_dir=str(tmp_path / "r8")))
    loaded = PromptArtifact.load(tmp_path / "r8" / "artifact.json")
    assert loaded.render() == a.render()
