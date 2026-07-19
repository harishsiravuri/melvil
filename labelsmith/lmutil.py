"""LM construction on top of dspy.LM (disk caching + per-call history with
usage/cost — both proven in the pilot). dspy is imported lazily so that setting
the cache directory works and importing labelsmith stays cheap.

Conventions (pilot-tested):
- task LM: temperature 0 -> identical prompts share the disk cache across runs
  and seeds (free repeats, deterministic).
- reflection LM: temperature 1.0 with ``rollout_id=seed`` — namespaces the
  cache per seed so different seeds never collapse onto one cached completion.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_CACHE = "DSPY_CACHEDIR"


def setup_cache(cache_dir: str | None = None) -> None:
    """Point the LM disk cache somewhere sensible (call before first LM use).
    No-op if the env var is already set."""
    if cache_dir:
        os.environ[_ENV_CACHE] = str(cache_dir)
    else:
        os.environ.setdefault(_ENV_CACHE, os.path.join(os.getcwd(), ".dspy_cache"))


def make_lm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    rollout_id: int | None = None,
    cache: bool = True,
) -> Any:
    setup_cache()
    import dspy

    kwargs: dict[str, Any] = {}
    if rollout_id is not None:
        kwargs["rollout_id"] = rollout_id
    return dspy.LM(model, temperature=temperature, max_tokens=max_tokens, cache=cache, **kwargs)


def text_of(lm_output: Any) -> str:
    """dspy.LM returns a list of str-or-dict; normalize to one string."""
    if isinstance(lm_output, list):
        lm_output = lm_output[0]
    if isinstance(lm_output, dict):
        lm_output = lm_output.get("text", "")
    return str(lm_output)


def reflection_callable(lm: Any):
    """Adapt a dspy-style LM to gepa's ``reflection_lm: str -> str`` protocol."""

    def call(prompt: str) -> str:
        return text_of(lm(prompt)).strip()

    return call
