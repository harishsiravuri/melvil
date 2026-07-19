# Contributing to labelsmith

Thanks for considering a contribution!

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest            # must stay green with NO API keys (fake-LM offline suite)
ruff check .
```

## Ground rules

- **No API keys in tests.** Anything touching an LLM in tests goes through
  `labelsmith.testing` fakes. If your feature can't be exercised offline,
  extend the fakes first.
- **The artifact is the contract.** Changes to `PromptArtifact` fields bump
  `schema_version` and keep `load()` backwards-compatible.
- **Budget honesty.** Any code path that spends metric calls must account for
  them in the artifact's `budget` block.
- Type hints on the public API; `ruff check .` clean; match the surrounding
  style.

## Where things live

See the module map in [docs/api.md](docs/api.md) and the architecture notes in
module docstrings — `adapter.py` (gepa integration), `features/` (the
classification layer), `proposer.py` (reflection prompting).

## Benchmarks

`benchmarks/` is a fixed matrix (datasets × arms × seeds) driven by the
library. If you claim a quality improvement, add or rerun the relevant arm and
include the numbers (and their cost) in the PR.
