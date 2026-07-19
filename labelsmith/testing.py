"""Fake LMs for offline testing — no API keys, no network, deterministic.

Salvaged from the pilot's offline harness. The core trick: `FakeTaskLM`'s
per-example accuracy *improves with prompt quality* (label definitions filled
in, boundary rules present, exemplars present), so the optimizer has a real
gradient to climb and accept/reject machinery is genuinely exercised — while
`FakeReflectionLM` produces component rewrites in the exact format the
proposer expects.

Public because downstream users can test their own pipelines without spending
money: build fakes with `fake_lms_for(...)` and route labelsmith's LM
construction to them with `patch_lms(monkeypatch, *fakes)` (works with the
pytest `monkeypatch` fixture, or pass the `labelsmith.lmutil` module for
manual patching).
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from labelsmith.data import Example


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class FakeLMBase:
    """Mimics the parts of dspy.LM that labelsmith uses: callable, `.model`,
    `.history` with usage dicts."""

    def __init__(self, model: str):
        self.model = model
        self.history: list[dict[str, Any]] = []

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.history.append(
            {
                "usage": {
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": 0.0,
                },
                "cost": 0.0,
            }
        )


class FakeTaskLM(FakeLMBase):
    """Deterministic classifier whose accuracy rises with prompt quality.

    Quality signal (max ~0.93): base 0.45
    + up to 0.30 for the fraction of labels with a non-trivial definition
    + 0.10 if boundary rules present + 0.05 if exemplars present
    + up to 0.03 with total prompt length.
    """

    def __init__(self, gold_map: dict[str, str], label_names: list[str]):
        super().__init__(model="fake/task")
        self.gold_map = gold_map
        self.label_names = label_names

    def _quality(self, system: str) -> float:
        q = 0.45
        defined = sum(
            1 for name in self.label_names
            if re.search(rf"- {re.escape(name)}: .{{25,}}", system)
        )
        q += 0.30 * defined / max(1, len(self.label_names))
        if "Boundary rules:" in system and len(system.split("Boundary rules:")[1]) > 60:
            q += 0.10
        if "Examples:" in system:
            q += 0.05
        q += min(len(system), 6000) / 200000
        return min(q, 0.93)

    def __call__(self, prompt: str | None = None, messages: list | None = None, **kw) -> list[str]:
        system = user = ""
        if messages:
            for m in messages:
                if m["role"] == "system":
                    system = str(m["content"])
                elif m["role"] == "user":
                    user = str(m["content"])
        else:
            user = str(prompt)
        self._record(len(system + user) // 4, 4)
        gold = self.gold_map.get(user)
        if gold is None:
            return [self.label_names[_h(user) % len(self.label_names)]]
        roll = _h(system + "|" + user) % 1000 / 1000.0
        if roll < self._quality(system):
            return [gold]
        wrong = [n for n in self.label_names if n != gold]
        return [wrong[_h(user) % len(wrong)]] if wrong else [gold]


class FakeReflectionLM(FakeLMBase):
    """Returns component rewrites in the proposer's tagged-block format (or a
    single plain fenced block for gepa's default proposer). Each rewrite is a
    plausible longer definition, so FakeTaskLM's quality score rises."""

    def __init__(self, seed: int = 0):
        super().__init__(model="fake/reflection")
        self.rng = random.Random(seed)
        self.n = 0

    def _new_text(self, comp: str) -> str:
        self.n += 1
        filler = " ".join(
            f"Decide by feature f{self.rng.randint(0, 99)} when ambiguous."
            for _ in range(3)
        )
        if comp.startswith("label::"):
            name = comp.split("::", 1)[1]
            return (
                f"Texts that clearly express {name}; the deciding feature is explicit "
                f"{name}-specific vocabulary. {filler}"
            )
        if comp == "boundary_rules":
            return f"If two categories seem to apply, pick the more specific one. {filler}"
        return f"Read carefully, weigh all categories, then answer. {filler}"

    def __call__(self, prompt: str | None = None, messages: list | None = None, **kw) -> list[str]:
        text = str(prompt if prompt is not None else messages)
        self._record(len(text) // 4, 120)
        comps = re.findall(r"```component:\s*([^\n`]+)", text)
        wanted = list(dict.fromkeys(c.strip() for c in comps))
        if wanted:
            blocks = [f"```component: {c}\n{self._new_text(c)}\n```" for c in wanted]
            return ["I analyzed the failures.\n\n" + "\n\n".join(blocks)]
        # default gepa proposer path: single fenced block = whole new instruction
        return [
            "Analysis of failures.\n```\nClassify the given text into exactly one category. "
            + self._new_text("instruction")
            + "\n```"
        ]


def fake_lms_for(
    train: list[Example], dev: list[Example], label_names: list[str], seed: int = 0
) -> tuple[FakeTaskLM, FakeReflectionLM]:
    gold = {e.text: e.label for e in [*train, *dev]}
    return FakeTaskLM(gold, label_names), FakeReflectionLM(seed)


def patch_lms(monkeypatch_or_module, task_lm: FakeTaskLM, reflection_lm: FakeReflectionLM):
    """Route labelsmith's LM construction to the fakes.

    Accepts a pytest `monkeypatch` fixture (preferred) or the
    `labelsmith.lmutil` module itself (manual patching)."""

    def fake_make_lm(model: str, temperature: float = 0.0, max_tokens: int = 512,
                     rollout_id: int | None = None, cache: bool = True):
        return reflection_lm if temperature > 0.5 else task_lm

    import importlib

    # NOTE: `labelsmith.optimize`/`labelsmith.evaluate` as *attributes* of the
    # package are the re-exported functions; go through importlib for modules.
    modules = [
        importlib.import_module(f"labelsmith.{m}")
        for m in ("lmutil", "optimize", "evaluate", "screen")
    ]
    for mod in modules:
        if hasattr(monkeypatch_or_module, "setattr"):
            monkeypatch_or_module.setattr(mod, "make_lm", fake_make_lm)
        else:
            mod.make_lm = fake_make_lm
    return fake_make_lm
