"""Classification-aware instruction proposer.

Replaces gepa's default single-component proposer when any classification
feature needs it. Differences from the default:
- can update SEVERAL components in one reflection call (both sides of a
  confused label boundary — coupled updates);
- optionally injects the dev confusion analysis block;
- component-aware guidance (a label definition is rewritten as a definition,
  not as a whole prompt);
- logs every reflection (prompt, response, proposal) to a JSONL trace file —
  the pilot's instrumentation discipline.

When ALL classification features are off, the adapter exposes
``propose_new_texts = None`` and gepa uses its stock proposer — that arm is
genuinely vanilla GEPA.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROPOSAL_TEMPLATE = """You are improving one part of a text-classification prompt. The prompt is assembled from named components (a task instruction, one definition per category, boundary rules). You will rewrite {n_components} component(s); the rest of the prompt stays fixed.

Current text of the component(s) to rewrite:
{current_components}

The full assembled prompt currently reads:
```
{full_prompt}
```

Examples of inputs, the assistant's responses under the current prompt, and feedback:
{examples_block}
{confusion_block}
Rewrite the listed component(s) to fix the observed failures. Requirements:
- A category definition must stay a compact definition of that single category (1-3 sentences): what belongs to it, and the deciding feature versus categories it gets confused with.
- Boundary rules must be short "if X vs Y, decide by Z" rules.
- The task instruction must stay general task guidance; it must not enumerate categories.
- Keep any factual, task-specific insight from the feedback — future inputs won't come with feedback.
- Do not change the output format contract (single category name); it is enforced elsewhere.

Return each rewritten component in its own fenced block, tagged with the component name, exactly like:
```component: {first_component}
<new text>
```
Return {n_components} block(s), nothing else."""


def _render_reflective_examples(records: Sequence[Mapping[str, Any]], cap: int = 6) -> str:
    parts = []
    for i, rec in enumerate(records[:cap]):
        inputs = rec.get("Inputs", {})
        outputs = rec.get("Generated Outputs", {})
        fb = rec.get("Feedback", "")
        text = inputs.get("text", "") if isinstance(inputs, dict) else str(inputs)
        raw = outputs.get("response", outputs) if isinstance(outputs, dict) else outputs
        parts.append(
            f"# Example {i + 1}\nText: {text}\nAssistant response: {raw}\nFeedback: {fb}"
        )
    return "\n\n".join(parts)


_BLOCK_RE = re.compile(r"```component:\s*(?P<name>[^\n`]+)\n(?P<body>.*?)```", re.DOTALL)


def parse_component_blocks(response: str, expected: list[str]) -> dict[str, str]:
    """Extract ```component: name ...``` blocks. Falls back to treating a single
    plain fenced block as the sole expected component."""
    out: dict[str, str] = {}
    for m in _BLOCK_RE.finditer(response):
        name = m.group("name").strip()
        if name in expected:
            out[name] = m.group("body").strip()
    if not out and len(expected) == 1:
        plain = re.findall(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
        if plain:
            out[expected[0]] = max(plain, key=len).strip()
    return out


class ClassificationProposer:
    """gepa ProposalFn. Built per-run by the adapter."""

    def __init__(
        self,
        render_full_prompt,  # () -> str, current candidate's assembled prompt
        confusion_block,  # () -> str ("" when confusion_reflection is off)
        trace_path: str | Path | None = None,
    ):
        self.render_full_prompt = render_full_prompt
        self.confusion_block = confusion_block
        self.trace_path = str(trace_path) if trace_path else None
        self.n_proposals = 0

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        call_lm = getattr(self, "_lm", None)
        if call_lm is None:
            raise RuntimeError("ClassificationProposer needs bind_lm() before use")

        current = "\n\n".join(
            f"```component: {name}\n{candidate[name]}\n```" for name in components_to_update
        )
        # merge reflective examples across the components being updated (dedup by text)
        seen: set[str] = set()
        merged: list[Mapping[str, Any]] = []
        for name in components_to_update:
            for rec in reflective_dataset.get(name, []):
                key = str(rec.get("Inputs", ""))[:300]
                if key not in seen:
                    seen.add(key)
                    merged.append(rec)
        confusion = self.confusion_block()
        prompt = PROPOSAL_TEMPLATE.format(
            n_components=len(components_to_update),
            current_components=current,
            full_prompt=self.render_full_prompt(candidate),
            examples_block=_render_reflective_examples(merged),
            confusion_block=f"\n{confusion}\n" if confusion else "",
            first_component=components_to_update[0],
        )
        response = call_lm(prompt)
        parsed = parse_component_blocks(response, components_to_update)
        result = {name: parsed.get(name, candidate[name]) for name in components_to_update}

        if self.trace_path:
            record = {
                "ts": time.time(),
                "proposal_idx": self.n_proposals,
                "components": components_to_update,
                "current": {n: candidate[n] for n in components_to_update},
                "reflection_prompt": prompt,
                "reflection_response": response,
                "proposed": result,
                "parsed_ok": sorted(parsed),
            }
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        self.n_proposals += 1
        return result

    def bind_lm(self, call_lm) -> None:
        """Bind the reflection LM (a str -> str callable)."""
        self._lm = call_lm
