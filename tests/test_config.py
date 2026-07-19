import pytest

from melvil import BUDGET_PRESETS, Config, Features


def make_config(**kw):
    return Config(task_model="openai/gpt-4.1-mini", reflection_model="openai/gpt-4.1", **kw)


def test_budget_presets():
    assert make_config(budget="light").metric_calls == BUDGET_PRESETS["light"]
    assert make_config(budget=123).metric_calls == 123
    with pytest.raises(ValueError, match="budget"):
        make_config(budget="huge").metric_calls  # noqa: B018


def test_features_toggles():
    f = Features.none()
    assert not (f.codebook or f.confusion_reflection or f.hard_example_mining)
    assert not Features().codebook  # default is vanilla GEPA (confirmation benchmarks)
    assert Features.all().codebook


def test_config_hash_stability_and_sensitivity():
    a, b = make_config(), make_config()
    assert a.config_hash == b.config_hash
    assert make_config(seed=1).config_hash != a.config_hash
    assert make_config(features=Features.all()).config_hash != a.config_hash
    # run_dir and thread count must NOT affect the hash
    assert make_config(run_dir="/tmp/x", num_threads=2).config_hash == a.config_hash


def test_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "task_model: openai/gpt-4.1-mini\n"
        "reflection_model: openai/gpt-4.1\n"
        "budget: medium\n"
        "features: {codebook: false, confusion_reflection: true, hard_example_mining: false}\n"
        "prices: {'my/model': [1.0, 2.0]}\n"
    )
    cfg = Config.from_yaml(p)
    assert cfg.metric_calls == BUDGET_PRESETS["medium"]
    assert cfg.features.confusion_reflection and not cfg.features.codebook
    assert cfg.prices["my/model"] == (1.0, 2.0)
