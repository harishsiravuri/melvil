# labelsmith API reference

Everything importable from `labelsmith` directly (`import labelsmith as ls`).
Model names throughout are LiteLLM ids (`openai/gpt-4.1-mini`,
`openrouter/openai/gpt-4.1`, ...).

## Data

### `Example(text: str, label: str)`
Frozen dataclass; one labeled example.

### `load_csv(path, text_col="text", label_col="label") -> list[Example]`
Load a CSV with header. Raises `ValueError` on missing columns or zero rows.

### `load_hf(dataset_id, text_field="text", label_field="label", config=None, split="train", limit=None) -> list[Example]`
Load from HuggingFace (`pip install labelsmith[hf]`). Integer class labels are
mapped to their string names. Script-based hub datasets are retried via their
`refs/convert/parquet` branch.

### `train_dev_split(examples, dev_size, seed=0) -> (train, dev)`
Stratified (largest-remainder, ≥1 per present class) split; deterministic per seed.

## TaskSpec

### `Label(name, description="", boundary_notes="", exemplars=[], auto_drafted=False)`

### `TaskSpec(name, labels: list[Label], instruction="")`
Validates that label names don't collide after canonicalization (lowercase,
space/hyphen → underscore). `instruction` overrides the default seed task
instruction.

- `TaskSpec.from_examples(name, examples)` — bare taxonomy from data.
- `TaskSpec.from_csv(path, name=None, text_col="text", label_col="label")`
- `TaskSpec.from_hf(dataset_id, ...)`
- `TaskSpec.from_yaml(path)` / `spec.to_yaml(path)` — full taxonomy with
  descriptions, boundary notes, exemplars.
- `spec.autodraft_descriptions(examples, model, k_per_label=5, overwrite=False) -> int`
  — LLM-draft empty descriptions (one call); drafted labels get
  `auto_drafted=True`.
- `spec.label_names`, `spec.label(name)`.

## Config

### `Features(codebook=True, confusion_reflection=True, hard_example_mining=True)`
The classification layer; each independently toggleable. `Features.none()` =
vanilla GEPA (single free-text instruction, stock proposer).

### `Config(task_model, reflection_model, budget="light", seed=0, features=Features(), num_threads=8, run_dir=None, ...)`
- `budget`: `"light"` (800) / `"medium"` (2000) / `"heavy"` (5000) metric
  calls, or an int. One metric call = one task-model classification.
- `task_temperature=0.0`, `task_max_tokens=40`, `reflection_temperature=1.0`,
  `reflection_max_tokens=8000`, `reflection_minibatch_size=3`,
  `max_exemplars=6`.
- `prices`: `{model_id: (usd_per_M_input, usd_per_M_output)}` override for the
  cost estimator/accounting.
- `config.metric_calls`, `config.config_hash` (stable across `run_dir` /
  `num_threads` changes), `Config.from_yaml(path)`.

## optimize

### `optimize(spec, train, dev, config, *, resume=False, on_round=None, parent=None) -> PromptArtifact`
Runs GEPA with the classification adapter. The dev set drives all optimization
decisions. Writes a run directory (default `runs/<task>/<confighash>-s<seed>/`):
`config.json`, gepa engine state (checkpoint), `traces.jsonl` (every
reflection), `artifact.json`.

- `resume=True`: return the saved artifact if the run finished, else continue
  from the engine checkpoint. Without it, an existing run directory raises
  `FileExistsError` (no silent clobber).
- `on_round(info: RoundInfo)`: called after every full dev evaluation.
  `RoundInfo(round, dev_score, best_dev_score, metric_calls_spent, cost_usd,
  top_confusions)`.
- `parent`: record lineage (`artifact.parent_id`).
- When `hard_example_mining` is on, one full dev evaluation is reserved from
  the budget for the exemplar accept/reject check.

## PromptArtifact

Versioned JSON (schema_version 1): components, label order, exemplars, models,
budget actually spent (metric calls, tokens, USD), dev scores + per-label
stats + confusion matrix, the dev-score-vs-budget curve, config snapshot +
hash, lineage, library version.

- `artifact.render() -> str` — the deployable prompt (identical to what was
  executed during optimization; ends with a fixed output-format contract).
- `artifact.diff(other) -> str` — score deltas + per-component unified diff.
- `artifact.save(path)` / `PromptArtifact.load(path)`.

Component names: `task_instruction`, `label::<name>`, `boundary_rules`
(codebook mode) or a single `instruction` (blob mode).

## evaluate

### `evaluate(artifact, data, model=None, num_threads=8, prices=None) -> Report`
Accuracy, macro-F1 (over labels with support), per-label P/R/F1/support,
confusion matrix (with an unparseable-output column), top confusions, measured
cost. `model` defaults to the artifact's task model; a different model makes it
a transfer evaluation (`report.transfer == True`).

### `report(report_or_artifact) -> str`
Markdown rendering for notebooks/docs.

## Costs

### `estimate_optimize_cost(spec, train, dev, config) -> CostEstimate`
Dry-run **upper bound** (ignores caching) — print it before spending.

### `estimate_evaluate_cost(artifact, data, model, prices=None) -> CostEstimate`

## Offline testing (`labelsmith.testing`)

`fake_lms_for(train, dev, label_names, seed=0)` returns `(FakeTaskLM,
FakeReflectionLM)` — deterministic fakes where task-LM accuracy rises with
prompt quality (real optimization gradient, zero cost). Route the library to
them with `patch_lms(monkeypatch, task_lm, reflection_lm)` (pytest) or
`patch_lms(labelsmith.lmutil, ...)` (manual). The library's own test suite
runs entirely on these.
