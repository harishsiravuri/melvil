# labelsmith

**Labeled examples + a label taxonomy in → an optimized, versioned classifier prompt out.**

labelsmith wraps the [GEPA](https://github.com/gepa-ai/gepa) reflective prompt-evolution
engine with a classification-specific layer that generic prompt optimizers lack:
confusion-driven reflection, a per-label prompt codebook, and hard-example mining.
It is a pure Python library — the API is the product.

```python
import labelsmith as ls

examples = ls.load_csv("tickets.csv")                  # text,label columns
train, dev = ls.train_dev_split(examples, dev_size=100, seed=0)
spec = ls.TaskSpec.from_examples("ticket-intents", examples)
cfg = ls.Config(task_model="openai/gpt-4.1-mini",
                reflection_model="openai/gpt-4.1", budget="light")
artifact = ls.optimize(spec, train, dev, cfg)
print(artifact.render())                               # deployable prompt string
artifact.save("ticket_intents.v1.json")
```

## Install

```bash
pip install -e .            # from a checkout
pip install -e '.[hf,dev]'  # + HuggingFace loaders + test/lint tooling
```

Model names are [LiteLLM](https://docs.litellm.ai) ids (`openai/...`,
`anthropic/...`, `openrouter/...`); set the matching API key env var
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, ...).

## What the benchmarks say (read this before choosing features)

We benchmark honestly, including against ourselves — full protocol and numbers in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md). The pre-registered confirmation pass
(8 public datasets, fresh seeds, light budget, gpt-4.1-mini) found:

- **Vanilla GEPA over labelsmith's rendered prompt is the strongest configuration**
  (mean test accuracy 0.781 vs 0.755 for the full classification layer and 0.762 for
  MIPROv2; seed prompt 0.703). At light budgets, prefer `features=ls.Features.none()`.
- The classification layer's per-component updates trade whole-prompt coverage for
  structure; at ~6–10 accepted proposals per run that trade loses, especially on hard
  small-taxonomy tasks. Whether it wins at medium/heavy budgets is an open question.
- Hard-example mining, behind its (strict, quarantined) accept gate, kept exemplars in
  0/24 confirmation runs — treat it as a safety-gated no-op at light budgets.

## What the classification layer does

Each feature is independently toggleable via `ls.Features` (all on by default;
`Features.none()` gives you vanilla GEPA):

1. **Per-label codebook** — the prompt is not a free-text blob but named
   components: a task instruction, one definition per label, and boundary
   rules. GEPA evolves them per-component.
2. **Confusion-driven reflection** — every full dev evaluation updates a
   confusion matrix; reflection rounds are pointed at the label components on
   both sides of the currently worst confused boundary, and the reflection LM
   is shown the top confused pairs with concrete misclassified examples.
3. **Hard-example mining** — dev examples that stay misclassified across the
   run become candidate few-shot exemplars, selected to cover the top confused
   boundaries, and kept only if they don't hurt dev accuracy. (Exemplars come
   from dev, so dev scores of exemplar-augmented artifacts are mildly
   optimistic — judge them on test.)

The rendered prompt always ends with a fixed, non-evolvable output-format
contract, so the optimizer can never break parseability.

## Everything else you get

- **`PromptArtifact`** — versioned JSON: components, models, budget and cost
  actually spent, dev scores + confusion matrix, the dev-score-vs-budget curve,
  config hash, lineage (`parent_id`). `artifact.diff(other)` gives a
  per-component diff with score deltas.
- **`evaluate(artifact, data, model=...)`** — accuracy, macro-F1, per-label
  P/R/F1, confusion, cost; pass a different `model` for a transfer evaluation.
  `report(...)` renders it as markdown.
- **Cost estimation before spending** — `estimate_optimize_cost(spec, train,
  dev, cfg)` is a dry-run upper bound; measured spend comes from the LM call
  history and lands in the artifact.
- **Run directories & resume** — every run writes `runs/<task>/<hash>-s<seed>/`
  (config, engine state, reflection traces, artifact);
  `optimize(..., resume=True)` continues an interrupted run.
- **Progress** — pass `on_round=lambda info: ...` for live dev score / spend
  after every full dev evaluation, or just read the default logging.
- **Offline testing** — `labelsmith.testing` ships fake LMs with a real
  optimization gradient; the whole test suite runs with no API keys.

## Worked example & docs

- [examples/quickstart.py](examples/quickstart.py) — the snippet above, runnable end-to-end on AG News.
- [examples/agnews_demo.ipynb](examples/agnews_demo.ipynb) — notebook walkthrough with a live progress callback.
- [docs/api.md](docs/api.md) — public API reference.
- [benchmarks/](benchmarks/) — the honest-benchmark harness (datasets × arms × seeds).

## Development

```bash
pip install -e '.[dev]'
pytest          # green with no API keys — fake-LM offline suite
ruff check .
```

License: MIT.
