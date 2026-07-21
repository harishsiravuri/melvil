"""Cost accounting (measured, from lm.history — pilot-proven) and dry-run
estimation (upper bound, printed before any paid run)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: USD per 1M tokens: (input, output). Extend/override via Config.prices.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic/claude-sonnet-4-5-20250929": (3.00, 15.00),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.03),
}


def price_for(
    model: str, prices: dict[str, tuple[float, float]] | None = None
) -> tuple[float, float]:
    table = {**DEFAULT_PRICES, **(prices or {})}
    if model in table:
        return table[model]
    # tolerate provider prefixes, e.g. "openrouter/openai/gpt-4.1"
    for key, val in table.items():
        if model.endswith(key) or key.endswith(model):
            return val
    raise KeyError(
        f"no price known for model {model!r}; "
        f"pass Config(prices={{{model!r}: (in_per_M, out_per_M)}})"
    )


def lm_usage(lm: Any, since: int = 0) -> dict[str, Any]:
    """Sum real (non-cache-hit) usage from lm.history[since:]. Thread-safe by
    construction (history is per-instance and append-only); includes provider-
    reported per-call cost when available (OpenRouter reports it)."""
    pt = ct = calls = hits = 0
    cost = 0.0
    for e in lm.history[since:]:
        u = e.get("usage") or {}
        if not u.get("total_tokens"):
            hits += 1
            continue
        calls += 1
        pt += u.get("prompt_tokens") or 0
        ct += u.get("completion_tokens") or 0
        cost += u.get("cost") or e.get("cost") or 0.0
    return {
        "model": getattr(lm, "model", "?"),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "calls": calls,
        "cache_hits": hits,
        "cost_usd": round(float(cost), 6),
    }


def usage_cost_usd(
    usage: dict[str, Any], prices: dict[str, tuple[float, float]] | None = None
) -> float:
    """Cost of a usage record; prefers provider-reported cost, falls back to the
    price table."""
    if usage.get("cost_usd"):
        return float(usage["cost_usd"])
    try:
        pin, pout = price_for(usage.get("model", ""), prices)
    except KeyError:
        return 0.0
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)  # chars/4 heuristic


@dataclass
class CostEstimate:
    total_usd: float
    lines: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        body = "\n".join(f"  {ln}" for ln in self.lines)
        return f"{body}\n  ESTIMATED TOTAL (upper bound): ${self.total_usd:.2f}"


def estimate_optimize_cost(spec, train, dev, config) -> CostEstimate:
    """Upper-bound dry-run estimate for one `optimize()` call. Assumptions
    (calibrated on the pilot): the evolved prompt is ~3x the seed prompt;
    ~1 reflection per 60 metric calls with ~3.5k in / 700 out tokens; caching
    is ignored (it only lowers real spend)."""
    from melvil.artifact import render_prompt
    from melvil.optimize import seed_candidate_for

    candidate = seed_candidate_for(spec)
    seed_prompt = render_prompt(candidate, spec.label_names)
    avg_text = sum(_tokens(e.text) for e in (train + dev)[:200]) / max(1, len((train + dev)[:200]))
    per_call_in = 3 * _tokens(seed_prompt) + avg_text + 20
    per_call_out = 8
    n_calls = config.metric_calls
    tin, tout = price_for(config.task_model, config.prices)
    rin, rout = price_for(config.reflection_model, config.prices)
    task_usd = n_calls * (per_call_in * tin + per_call_out * tout) / 1e6
    n_refl = max(4, n_calls // 60)
    refl_usd = n_refl * (3500 * rin + 700 * rout) / 1e6
    total = task_usd + refl_usd
    return CostEstimate(
        total_usd=total,
        lines=[
            f"task model {config.task_model}: {n_calls} calls x "
            f"~{per_call_in:.0f} in / {per_call_out} out tok = ${task_usd:.2f}",
            f"reflection {config.reflection_model}: ~{n_refl} calls x "
            f"~3.5k/700 tok = ${refl_usd:.2f}",
        ],
    )


def estimate_draft_cost(spec, dev, config, iterations: int = 2) -> CostEstimate:
    """Upper-bound dry-run for one `draft()` call: (iterations+1) dev
    evaluations + `iterations` reflection calls (diagnosis + full rewrite)."""
    from melvil.artifact import render_prompt
    from melvil.optimize import seed_candidate_for

    seed_prompt = render_prompt(seed_candidate_for(spec), spec.label_names)
    avg_text = sum(_tokens(e.text) for e in dev[:200]) / max(1, len(dev[:200]))
    per_call_in = 2 * _tokens(seed_prompt) + avg_text + 20
    tin, tout = price_for(config.task_model, config.prices)
    rin, rout = price_for(config.reflection_model, config.prices)
    evals_usd = (iterations + 1) * len(dev) * (per_call_in * tin + 8 * tout) / 1e6
    refl_usd = iterations * ((2000 + 2 * _tokens(seed_prompt)) * rin + 900 * rout) / 1e6
    return CostEstimate(
        total_usd=evals_usd + refl_usd,
        lines=[
            f"{iterations + 1} dev evals x {len(dev)} calls = ${evals_usd:.2f}",
            f"{iterations} diagnosis rewrites ({config.reflection_model}) = ${refl_usd:.2f}",
        ],
    )


def estimate_evaluate_cost(artifact, data, model: str, prices=None) -> CostEstimate:
    per_call_in = _tokens(artifact.render()) + 40
    pin, pout = price_for(model, prices)
    usd = len(data) * (per_call_in * pin + 8 * pout) / 1e6
    return CostEstimate(usd, [f"{model}: {len(data)} calls x ~{per_call_in} in tok = ${usd:.2f}"])
