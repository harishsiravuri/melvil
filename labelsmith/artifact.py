"""PromptArtifact: the versioned output of an optimization run.

An artifact is a JSON document holding the evolved prompt *components*, the
models/budget/scores/config that produced them, and lineage. `render()`
assembles the deployable prompt string — the exact prompt that was executed
during optimization, not a reconstruction.

Component naming convention:
- codebook mode: ``task_instruction``, ``label::<name>`` (one per label),
  ``boundary_rules``
- blob mode (codebook off): a single ``instruction`` component
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labelsmith._about import LIBRARY_NAME, __version__

SCHEMA_VERSION = 1

TASK_COMPONENT = "task_instruction"
BLOB_COMPONENT = "instruction"
BOUNDARY_COMPONENT = "boundary_rules"
LABEL_PREFIX = "label::"

#: Fixed output contract appended to every rendered prompt. It is NOT an
#: evolvable component, so the optimizer can never break the output format.
OUTPUT_CONTRACT = "Respond with exactly one category name from the list above and nothing else."


def label_component(name: str) -> str:
    return f"{LABEL_PREFIX}{name}"


def render_prompt(
    components: dict[str, str],
    label_names: list[str],
    exemplars: list[dict[str, str]] | None = None,
) -> str:
    """Assemble the deployable system prompt from components (+ optional mined
    few-shot exemplars). Used identically at optimize time and via
    `PromptArtifact.render()`."""
    parts: list[str] = []
    if BLOB_COMPONENT in components:  # blob mode
        parts.append(components[BLOB_COMPONENT].strip())
    else:
        parts.append(components.get(TASK_COMPONENT, "").strip())
        defs = []
        for name in label_names:
            desc = components.get(label_component(name), "").strip()
            defs.append(f"- {name}: {desc}" if desc else f"- {name}")
        parts.append("Categories:\n" + "\n".join(defs))
        boundary = components.get(BOUNDARY_COMPONENT, "").strip()
        if boundary:
            parts.append("Boundary rules:\n" + boundary)
    if exemplars:
        shot = []
        for ex in exemplars:
            shot.append(f"Text: {ex['text']}\nLabel: {ex['label']}")
        parts.append("Examples:\n\n" + "\n\n".join(shot))
    parts.append(OUTPUT_CONTRACT)
    return "\n\n".join(p for p in parts if p)


@dataclass
class PromptArtifact:
    task_name: str
    components: dict[str, str]
    label_names: list[str]
    exemplars: list[dict[str, str]] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)  # metric_calls_spent, cost_usd, ...
    scores: dict[str, Any] = field(default_factory=dict)  # dev_accuracy, dev_macro_f1, ...
    confusion: dict[str, Any] | None = None  # {"labels": [...], "matrix": [[...]]} on dev
    curve: list[dict[str, Any]] = field(default_factory=list)  # metric_calls -> best dev score
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    parent_id: str | None = None
    artifact_id: str = ""
    created_at: str = ""
    library: dict[str, str] = field(
        default_factory=lambda: {"name": LIBRARY_NAME, "version": __version__}
    )
    schema_version: int = SCHEMA_VERSION
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if not self.artifact_id:
            payload = json.dumps(
                [self.task_name, self.components, self.exemplars, self.created_at,
                 self.config_hash],
                sort_keys=True,
            )
            self.artifact_id = hashlib.sha256(payload.encode()).hexdigest()[:12]

    # ------------------------------------------------------------- behavior
    def render(self) -> str:
        """The deployable prompt string."""
        return render_prompt(self.components, self.label_names, self.exemplars)

    def diff(self, other: PromptArtifact) -> str:
        """Human-readable comparison: per-component text diff + score deltas."""
        lines: list[str] = [f"artifact {other.artifact_id} -> {self.artifact_id}"]
        for key in ("dev_accuracy", "dev_macro_f1", "test_accuracy"):
            a, b = other.scores.get(key), self.scores.get(key)
            if a is not None or b is not None:
                def fmt(v):
                    return "—" if v is None else f"{v:.3f}"

                delta = f" ({b - a:+.3f})" if (a is not None and b is not None) else ""
                lines.append(f"  {key}: {fmt(a)} -> {fmt(b)}{delta}")
        all_comps = sorted(set(self.components) | set(other.components))
        for comp in all_comps:
            old = other.components.get(comp, "")
            new = self.components.get(comp, "")
            if old == new:
                continue
            lines.append(f"\n## {comp}")
            diff = difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="",
                fromfile=f"{comp}@{other.artifact_id}", tofile=f"{comp}@{self.artifact_id}",
            )
            lines.extend(list(diff)[2:] or ["(whitespace-only change)"])
        if len(all_comps) and len(lines) == 1:
            lines.append("  (no component changes)")
        ex_a, ex_b = len(other.exemplars), len(self.exemplars)
        if ex_a != ex_b:
            lines.append(f"\nexemplars: {ex_a} -> {ex_b}")
        return "\n".join(lines)

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "parent_id": self.parent_id,
            "task_name": self.task_name,
            "created_at": self.created_at,
            "library": self.library,
            "models": self.models,
            "budget": self.budget,
            "scores": self.scores,
            "config_hash": self.config_hash,
            "config": self.config,
            "label_names": self.label_names,
            "components": self.components,
            "exemplars": self.exemplars,
            "confusion": self.confusion,
            "curve": self.curve,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> PromptArtifact:
        d = json.loads(Path(path).read_text())
        if d.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema {d['schema_version']} is newer than this "
                f"{LIBRARY_NAME} ({SCHEMA_VERSION}); please upgrade"
            )
        return cls(
            task_name=d["task_name"],
            components=d["components"],
            label_names=d["label_names"],
            exemplars=d.get("exemplars", []),
            models=d.get("models", {}),
            budget=d.get("budget", {}),
            scores=d.get("scores", {}),
            confusion=d.get("confusion"),
            curve=d.get("curve", []),
            config=d.get("config", {}),
            config_hash=d.get("config_hash", ""),
            parent_id=d.get("parent_id"),
            artifact_id=d["artifact_id"],
            created_at=d.get("created_at", ""),
            library=d.get("library", {}),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            notes=d.get("notes", ""),
        )
