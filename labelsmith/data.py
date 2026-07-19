"""Plain data containers and loaders: CSV, HuggingFace, stratified splitting.

Salvaged from the pilot's data prep (stratified largest-remainder sampling,
script-dataset parquet fallback, canonical label matching).
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One labeled example."""

    text: str
    label: str


def canon(s: str) -> str:
    """Canonical form used for label comparison everywhere in the library."""
    return s.strip().lower().strip("'\".` ").replace(" ", "_").replace("-", "_")


def load_csv(path: str | Path, text_col: str = "text", label_col: str = "label") -> list[Example]:
    """Load examples from a CSV with (by default) `text` and `label` columns."""
    out: list[Example] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if cols is None or text_col not in cols or label_col not in cols:
            raise ValueError(
                f"CSV must have '{text_col}' and '{label_col}' columns; found {cols}"
            )
        for row in reader:
            if row[text_col] and row[label_col]:
                out.append(Example(text=row[text_col], label=row[label_col]))
    if not out:
        raise ValueError(f"no examples loaded from {path}")
    return out


def load_hf(
    dataset_id: str,
    text_field: str = "text",
    label_field: str = "label",
    config: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> list[Example]:
    """Load examples from a HuggingFace dataset. Integer class labels are mapped
    to their string names via the dataset's features. Requires `datasets`
    (install extra: `labelsmith[hf]`). Script-based hub datasets are retried via
    their auto-converted parquet branch."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_hf requires the 'datasets' package: pip install labelsmith[hf]"
        ) from e

    args = (dataset_id, config) if config else (dataset_id,)
    try:
        ds = load_dataset(*args, split=split)
    except RuntimeError as e:
        if "no longer supported" not in str(e):
            raise
        ds = load_dataset(*args, split=split, revision="refs/convert/parquet")

    feature = ds.features.get(label_field)
    names = getattr(feature, "names", None)
    out: list[Example] = []
    for ex in ds:
        label = ex[label_field]
        if names is not None and isinstance(label, int):
            label = names[label]
        out.append(Example(text=str(ex[text_field]), label=str(label)))
        if limit is not None and len(out) >= limit:
            break
    return out


def stratified_sample(
    examples: list[Example], n: int, rng: random.Random, exclude: set[int] | None = None
) -> list[int]:
    """Proportional stratified sample (largest remainder, >=1 per present class
    when possible). Returns indices into `examples`."""
    exclude = exclude or set()
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        if i not in exclude:
            by_label[e.label].append(i)
    for lst in by_label.values():
        rng.shuffle(lst)
    total = sum(len(v) for v in by_label.values())
    if n > total:
        raise ValueError(f"asked for {n} examples but only {total} available")
    labels = sorted(by_label)
    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for lab in labels:
        exact = n * len(by_label[lab]) / total
        quotas[lab] = int(exact)
        remainders[lab] = exact - int(exact)
    for lab in labels:
        if quotas[lab] == 0 and n >= len(labels):
            quotas[lab] = 1
    while sum(quotas.values()) < n:
        candidates = [lab for lab in labels if quotas[lab] < len(by_label[lab])]
        lab = max(candidates, key=lambda c: remainders[c])
        quotas[lab] += 1
        remainders[lab] = -1.0
        if all(remainders[c] < 0 for c in candidates):
            for c in candidates:
                remainders[c] = 0.5
    while sum(quotas.values()) > n:
        lab = max(labels, key=lambda c: quotas[c])
        quotas[lab] -= 1
    out: list[int] = []
    for lab in labels:
        out.extend(by_label[lab][: quotas[lab]])
    rng.shuffle(out)
    return out


def train_dev_split(
    examples: list[Example], dev_size: int, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Stratified train/dev split: dev gets `dev_size` examples, train the rest."""
    rng = random.Random(seed)
    dev_idx = stratified_sample(examples, dev_size, rng)
    dev_set = set(dev_idx)
    train = [e for i, e in enumerate(examples) if i not in dev_set]
    dev = [examples[i] for i in dev_idx]
    return train, dev
