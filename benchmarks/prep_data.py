"""Prepare the six benchmark datasets: stratified train=150 / dev=100 / test=300.

Protocol (identical to the pilot study that preceded this library):
train+dev drawn disjointly from each dataset's original train split, test from
the original test split, proportional stratification, data seed 0. Subset
constants (banking77's 25 intents, clinc's 30, massive's 30) are fixed here.

Run: python benchmarks/prep_data.py    (downloads from HF; no API keys)
Output: benchmarks/data/{task}.json
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from labelsmith.data import Example, load_hf, stratified_sample

DATA_DIR = Path(__file__).parent / "data"
DATA_SEED = 0
N_TRAIN, N_DEV, N_TEST = 150, 100, 300

# CLINC150: 6 domains x 5 intents (from the CLINC paper's domain grouping)
CLINC_INTENTS = [
    "transfer", "balance", "freeze_account", "pay_bill", "report_fraud",
    "credit_score", "report_lost_card", "credit_limit", "rewards_balance", "apr",
    "book_flight", "book_hotel", "car_rental", "lost_luggage", "travel_suggestion",
    "recipe", "calories", "restaurant_reviews", "cook_time", "restaurant_reservation",
    "weather", "alarm", "timer", "calculator", "definition",
    "payday", "taxes", "pto_request", "direct_deposit", "meeting_schedule",
]

TREC_NAMES = {"ABBR": "abbreviation", "ENTY": "entity", "DESC": "description",
              "HUM": "human", "LOC": "location", "NUM": "numeric"}


def build(name: str, train_pool: list[Example], test_pool: list[Example]) -> None:
    rng = random.Random(DATA_SEED)
    train_idx = stratified_sample(train_pool, N_TRAIN, rng)
    dev_idx = stratified_sample(train_pool, N_DEV, rng, exclude=set(train_idx))
    test_idx = stratified_sample(test_pool, N_TEST, rng)
    payload = {
        "labels": sorted({e.label for e in train_pool}),
        "train": [train_pool[i].__dict__ for i in train_idx],
        "dev": [train_pool[i].__dict__ for i in dev_idx],
        "test": [test_pool[i].__dict__ for i in test_idx],
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / f"{name}.json").write_text(json.dumps(payload, indent=0))
    print(f"{name}: {len(payload['labels'])} labels | 150/100/300 written")


def prep_banking77() -> None:
    train = load_hf("mteb/banking77", label_field="label_text")
    test = load_hf("mteb/banking77", label_field="label_text", split="test")
    all_names = sorted({e.label for e in train})
    # pilot protocol: 25 of 77 label ids sampled with rng(0); mteb label ids
    # follow sorted(label_text) order
    keep = {all_names[i] for i in sorted(random.Random(DATA_SEED).sample(range(77), 25))}
    build("banking77", [e for e in train if e.label in keep], [e for e in test if e.label in keep])


def prep_ag_news() -> None:
    build("ag_news", load_hf("fancyzhx/ag_news"), load_hf("fancyzhx/ag_news", split="test"))


def prep_emotion() -> None:
    build(
        "emotion",
        load_hf("dair-ai/emotion", config="split"),
        load_hf("dair-ai/emotion", config="split", split="test"),
    )


def prep_trec() -> None:
    def rename(examples: list[Example]) -> list[Example]:
        return [Example(e.text, TREC_NAMES[e.label]) for e in examples]

    build(
        "trec",
        rename(load_hf("CogComp/trec", label_field="coarse_label")),
        rename(load_hf("CogComp/trec", label_field="coarse_label", split="test")),
    )


def prep_clinc150() -> None:
    train = load_hf("clinc/clinc_oos", label_field="intent", config="plus")
    test = load_hf("clinc/clinc_oos", label_field="intent", config="plus", split="test")
    keep = set(CLINC_INTENTS)
    missing = keep - {e.label for e in train}
    assert not missing, f"clinc intents missing: {missing}"
    build("clinc150", [e for e in train if e.label in keep], [e for e in test if e.label in keep])


MASSIVE_PARQUET = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "refs%2Fconvert%2Fparquet/en-US/{split}/0000.parquet"
)

# The parquet conversion branch preserves the intent ClassLabel names, so ids
# map straight to names.
def _load_massive(split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={split: MASSIVE_PARQUET.format(split=split)},
                      split=split)
    feature = ds.features.get("intent")
    names = getattr(feature, "names", None)
    out = []
    for ex in ds:
        label = ex["intent"]
        if names is not None and isinstance(label, int):
            label = names[label]
        out.append(Example(text=str(ex["utt"]), label=str(label)))
    return out


def prep_massive() -> None:
    train = _load_massive("train")
    test = _load_massive("test")
    top30 = {lab for lab, _ in Counter(e.label for e in train).most_common(30)}
    build("massive", [e for e in train if e.label in top30], [e for e in test if e.label in top30])


def load_task(name: str) -> dict:
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    for split in ("train", "dev", "test"):
        d[split] = [Example(**e) for e in d[split]]
    return d


TASKS = ["banking77", "ag_news", "emotion", "trec", "clinc150", "massive"]

if __name__ == "__main__":
    prep_banking77()
    prep_ag_news()
    prep_emotion()
    prep_trec()
    prep_clinc150()
    prep_massive()
