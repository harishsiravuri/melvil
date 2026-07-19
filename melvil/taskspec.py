"""TaskSpec: the label taxonomy — names, descriptions, boundary notes, exemplars.

Loaders: from_examples / from_csv (text,label columns), from_hf (HuggingFace
dataset id), from_yaml. Descriptions can be auto-drafted from examples via LLM
(`autodraft_descriptions`); auto-drafted text is flagged as such.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from melvil.data import Example, canon, load_csv, load_hf


@dataclass
class Label:
    name: str
    description: str = ""
    boundary_notes: str = ""
    exemplars: list[Example] = field(default_factory=list)
    auto_drafted: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.boundary_notes:
            d["boundary_notes"] = self.boundary_notes
        if self.exemplars:
            d["exemplars"] = [{"text": e.text, "label": e.label} for e in self.exemplars]
        if self.auto_drafted:
            d["auto_drafted"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Label:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            boundary_notes=d.get("boundary_notes", ""),
            exemplars=[Example(**e) for e in d.get("exemplars", [])],
            auto_drafted=bool(d.get("auto_drafted", False)),
        )


@dataclass
class TaskSpec:
    """A classification task: a name, a label taxonomy, and (optionally) a
    task-level instruction. If `instruction` is empty a sensible default is
    used as the optimization seed."""

    name: str
    labels: list[Label]
    instruction: str = ""

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("TaskSpec needs at least one label")
        canons = [canon(lb.name) for lb in self.labels]
        dupes = {c for c in canons if canons.count(c) > 1}
        if dupes:
            raise ValueError(f"label names collide after canonicalization: {sorted(dupes)}")

    @property
    def label_names(self) -> list[str]:
        return [lb.name for lb in self.labels]

    def label(self, name: str) -> Label:
        c = canon(name)
        for lb in self.labels:
            if canon(lb.name) == c:
                return lb
        raise KeyError(name)

    # ------------------------------------------------------------- loaders
    @classmethod
    def from_examples(cls, name: str, examples: list[Example]) -> TaskSpec:
        """Build a bare taxonomy (names only) from labeled examples."""
        seen: dict[str, str] = {}
        for e in examples:
            seen.setdefault(canon(e.label), e.label)
        return cls(name=name, labels=[Label(name=v) for v in sorted(seen.values())])

    @classmethod
    def from_csv(
        cls, path: str | Path, name: str | None = None,
        text_col: str = "text", label_col: str = "label",
    ) -> TaskSpec:
        examples = load_csv(path, text_col=text_col, label_col=label_col)
        return cls.from_examples(name or Path(path).stem, examples)

    @classmethod
    def from_hf(
        cls, dataset_id: str, name: str | None = None,
        text_field: str = "text", label_field: str = "label",
        config: str | None = None, split: str = "train", limit: int | None = 2000,
    ) -> TaskSpec:
        examples = load_hf(
            dataset_id, text_field, label_field, config=config, split=split, limit=limit
        )
        return cls.from_examples(name or dataset_id.split("/")[-1], examples)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        return cls(
            name=d["name"],
            labels=[Label.from_dict(lb) for lb in d["labels"]],
            instruction=d.get("instruction", ""),
        )

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        d: dict[str, Any] = {"name": self.name, "labels": [lb.to_dict() for lb in self.labels]}
        if self.instruction:
            d["instruction"] = self.instruction
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))

    # ------------------------------------------------------------ drafting
    def autodraft_descriptions(
        self, examples: list[Example], model: str, k_per_label: int = 5, overwrite: bool = False,
    ) -> int:
        """Fill empty label descriptions by asking an LLM, using up to
        `k_per_label` examples per label. Drafted labels get auto_drafted=True.
        Returns the number of labels drafted. Costs one LM call."""
        from melvil.lmutil import make_lm, text_of

        targets = [lb for lb in self.labels if overwrite or not lb.description]
        if not targets:
            return 0
        by_label: dict[str, list[str]] = defaultdict(list)
        for e in examples:
            c = canon(e.label)
            if len(by_label[c]) < k_per_label:
                by_label[c].append(e.text)
        blocks = []
        for lb in targets:
            exs = "\n".join(f"  - {t[:200]}" for t in by_label.get(canon(lb.name), []))
            blocks.append(f"LABEL: {lb.name}\nEXAMPLES:\n{exs or '  (none available)'}")
        prompt = (
            "You are documenting a text-classification taxonomy. For each label below, "
            "write a one-sentence description of what texts belong to it, grounded in the "
            "example texts. Be specific enough to distinguish it from sibling labels.\n\n"
            + "\n\n".join(blocks)
            + "\n\nRespond with one line per label, exactly in the format:\n"
            "LABEL_NAME: description"
        )
        lm = make_lm(model, temperature=0.2, max_tokens=2000)
        out = text_of(lm(prompt))
        drafted = 0
        wanted = {canon(lb.name): lb for lb in targets}
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, _, desc = line.partition(":")
            lb = wanted.get(canon(key))
            if lb is not None and desc.strip():
                lb.description = desc.strip()
                lb.auto_drafted = True
                drafted += 1
        return drafted

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "labels": [lb.to_dict() for lb in self.labels],
        }
