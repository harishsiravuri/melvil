import pytest

from labelsmith import Config, TaskSpec, estimate_optimize_cost
from labelsmith.costs import lm_usage, price_for, usage_cost_usd
from labelsmith.data import Example
from labelsmith.testing import FakeTaskLM


def test_price_for_handles_provider_prefixes():
    assert price_for("openai/gpt-4.1-mini") == (0.40, 1.60)
    assert price_for("openrouter/openai/gpt-4.1-mini") == (0.40, 1.60)
    with pytest.raises(KeyError, match="no price known"):
        price_for("unknown/model")
    assert price_for("my/model", {"my/model": (1.0, 2.0)}) == (1.0, 2.0)


def test_lm_usage_counts_calls_and_cache_hits():
    lm = FakeTaskLM({"x": "a"}, ["a", "b"])
    lm(messages=[{"role": "system", "content": "a long system prompt " * 5},
                 {"role": "user", "content": "x"}])
    lm.history.append({"usage": {}, "cost": 0.5})  # simulated cache hit
    u = lm_usage(lm)
    assert u["calls"] == 1 and u["cache_hits"] == 1
    assert u["prompt_tokens"] > 0


def test_usage_cost_falls_back_to_price_table():
    u = {"model": "openai/gpt-4.1-mini", "prompt_tokens": 1_000_000, "completion_tokens": 0,
         "cost_usd": 0.0}
    assert usage_cost_usd(u) == pytest.approx(0.40)
    u2 = {**u, "cost_usd": 1.23}  # provider-reported wins
    assert usage_cost_usd(u2) == 1.23


def test_estimate_optimize_cost_scales_with_budget():
    spec = TaskSpec.from_examples("t", [Example("a text", "x"), Example("b text", "y")])
    data = [Example(f"sample {i}", "x") for i in range(20)]
    cfg_small = Config("openai/gpt-4.1-mini", "openai/gpt-4.1", budget=200)
    cfg_big = Config("openai/gpt-4.1-mini", "openai/gpt-4.1", budget=2000)
    e_small = estimate_optimize_cost(spec, data, data, cfg_small)
    e_big = estimate_optimize_cost(spec, data, data, cfg_big)
    assert 0 < e_small.total_usd < e_big.total_usd
    assert "ESTIMATED TOTAL" in str(e_small)
