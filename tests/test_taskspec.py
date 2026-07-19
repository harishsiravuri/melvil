import pytest

from labelsmith import Example, Label, TaskSpec
from labelsmith.data import canon, load_csv, train_dev_split


def make_examples(n_per=4):
    labels = ["billing", "shipping", "returns"]
    return [
        Example(text=f"{lab} question number {i}", label=lab)
        for lab in labels
        for i in range(n_per)
    ]


def test_from_examples_builds_sorted_taxonomy():
    spec = TaskSpec.from_examples("support", make_examples())
    assert spec.label_names == ["billing", "returns", "shipping"]
    assert all(not lb.description for lb in spec.labels)


def test_duplicate_canonical_labels_rejected():
    with pytest.raises(ValueError, match="collide"):
        TaskSpec(name="x", labels=[Label("My Label"), Label("my_label")])


def test_empty_labels_rejected():
    with pytest.raises(ValueError):
        TaskSpec(name="x", labels=[])


def test_label_lookup_is_canonical():
    spec = TaskSpec.from_examples("support", make_examples())
    assert spec.label("BILLING").name == "billing"
    with pytest.raises(KeyError):
        spec.label("nope")


def test_from_csv_and_loader(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\nhello,greet\nbye,farewell\n\"a, b\",greet\n")
    examples = load_csv(p)
    assert len(examples) == 3
    assert examples[2].text == "a, b"
    spec = TaskSpec.from_csv(p)
    assert spec.name == "d"
    assert spec.label_names == ["farewell", "greet"]


def test_csv_missing_columns(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("body,tag\nx,y\n")
    with pytest.raises(ValueError, match="columns"):
        load_csv(p)


def test_yaml_roundtrip(tmp_path):
    spec = TaskSpec(
        name="t",
        labels=[
            Label("a", description="letter a", boundary_notes="not b"),
            Label("b", exemplars=[Example("bee", "b")]),
        ],
        instruction="Custom instruction.",
    )
    path = tmp_path / "spec.yaml"
    spec.to_yaml(path)
    loaded = TaskSpec.from_yaml(path)
    assert loaded.name == "t"
    assert loaded.instruction == "Custom instruction."
    assert loaded.label("a").boundary_notes == "not b"
    assert loaded.label("b").exemplars[0].text == "bee"
    assert not loaded.label("a").auto_drafted


def test_train_dev_split_stratified_and_disjoint():
    examples = make_examples(n_per=10)
    train, dev = train_dev_split(examples, dev_size=9, seed=0)
    assert len(dev) == 9 and len(train) == 21
    dev_labels = [e.label for e in dev]
    assert all(dev_labels.count(lab) == 3 for lab in ("billing", "shipping", "returns"))
    assert {e.text for e in train}.isdisjoint({e.text for e in dev})


def test_canon():
    assert canon("  Card-Arrival ") == "card_arrival"
    assert canon('"Sci/Tech"') == "sci/tech"
