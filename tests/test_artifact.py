from melvil import PromptArtifact
from melvil.artifact import (
    BLOB_COMPONENT,
    OUTPUT_CONTRACT,
    TASK_COMPONENT,
    label_component,
    render_prompt,
)


def make_artifact(**kw):
    defaults = dict(
        task_name="support",
        components={
            TASK_COMPONENT: "Classify the ticket.",
            label_component("billing"): "Money things.",
            label_component("shipping"): "",
        },
        label_names=["billing", "shipping"],
        scores={"dev_accuracy": 0.8, "dev_macro_f1": 0.75},
    )
    defaults.update(kw)
    return PromptArtifact(**defaults)


def test_render_codebook_layout():
    text = make_artifact().render()
    assert "Classify the ticket." in text
    assert "- billing: Money things." in text
    assert "- shipping" in text and "- shipping:" not in text  # empty desc -> bare name
    assert text.rstrip().endswith(OUTPUT_CONTRACT)
    assert "Boundary rules:" not in text  # empty boundary component omitted


def test_render_blob_mode_and_exemplars():
    a = PromptArtifact(
        task_name="t",
        components={BLOB_COMPONENT: "Pick a label from: x, y."},
        label_names=["x", "y"],
        exemplars=[{"text": "example one", "label": "x"}],
    )
    text = a.render()
    assert text.startswith("Pick a label from: x, y.")
    assert "Categories:" not in text
    assert "Text: example one\nLabel: x" in text


def test_ids_and_lineage():
    a = make_artifact()
    b = make_artifact(parent_id=a.artifact_id, components={TASK_COMPONENT: "Better."})
    assert a.artifact_id and len(a.artifact_id) == 12
    assert b.parent_id == a.artifact_id


def test_save_load_roundtrip(tmp_path):
    a = make_artifact(
        curve=[{"metric_calls": 100, "dev_score": 0.6}],
        confusion={"labels": ["billing", "shipping"], "matrix": [[3, 1, 0], [0, 4, 0]]},
    )
    path = a.save(tmp_path / "a.json")
    b = PromptArtifact.load(path)
    assert b.artifact_id == a.artifact_id
    assert b.components == a.components
    assert b.render() == a.render()
    assert b.curve == a.curve
    assert b.confusion == a.confusion


def test_diff_reports_components_and_scores():
    a = make_artifact()
    b = make_artifact(
        components={**a.components, label_component("billing"): "Anything about invoices."},
        scores={"dev_accuracy": 0.9, "dev_macro_f1": 0.85},
    )
    d = b.diff(a)
    assert "dev_accuracy: 0.800 -> 0.900 (+0.100)" in d
    assert "label::billing" in d
    assert "-Money things." in d and "+Anything about invoices." in d
    assert TASK_COMPONENT not in d.split("label::billing")[1]  # unchanged comp not diffed


def test_render_prompt_matches_artifact_render():
    a = make_artifact()
    assert a.render() == render_prompt(a.components, a.label_names, a.exemplars)
