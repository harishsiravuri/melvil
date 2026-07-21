# melvil

**Labeled examples + a label taxonomy in → an optimized, versioned classifier prompt out.**

One error-grounded rewrite captures most of what full prompt evolution delivers, at
about a tenth of the cost — 60–91% of GEPA's gain in our frozen-protocol benchmarks
(model-family dependent), beating full GEPA outright on 2 of 8 tasks
([the numbers](mechanistic/RESULTS.md)). That is what `mv.draft()` does, and it is
where melvil starts:

```python
import melvil as mv

examples = mv.load_csv("tickets.csv")                  # text,label columns
train, dev = mv.train_dev_split(examples, dev_size=100, seed=0)
spec = mv.TaskSpec.from_examples("ticket-intents", examples)
cfg = mv.Config(task_model="openai/gpt-4.1-mini",
                reflection_model="openai/gpt-4.1")
artifact = mv.draft(spec, train, dev, cfg)             # diagnose errors -> write the prompt (x2)
print(artifact.render())                               # deployable prompt string
artifact.save("ticket_intents.v1.json")
```

`draft()` evaluates your seed prompt once on dev, builds a structured error diagnosis
(per-label accuracy, top confused pairs with concrete examples), has the reflection
model write a complete replacement prompt, and repeats once (`iterations=2`, the
benchmarked arm). The diagnosis reports are saved in the run directory. Full GEPA
evolution (`mv.optimize()`) remains fully supported — as the escalation path.

*Named for [Melvil Dewey](https://en.wikipedia.org/wiki/Melvil_Dewey), who gave
libraries a system for putting things in the right category.*

## Install

```bash
pip install pymelvil             # the PyPI distribution is `pymelvil`; you `import melvil`
pip install 'pymelvil[hf]'       # + HuggingFace dataset loaders
```

From a checkout:

```bash
pip install -e '.[hf,dev]'       # + test/lint tooling
```

Model names are [LiteLLM](https://docs.litellm.ai) ids (`openai/...`,
`anthropic/...`, `openrouter/...`); set the matching API key env var
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, ...).

## When to run full optimization

The recommended workflow is **draft → screen → maybe optimize**:

```python
artifact = mv.draft(spec, train, dev, cfg)
check = mv.screen(artifact, dev, cfg)      # remaining headroom AFTER drafting
if check.verdict == "headroom":
    better = mv.optimize(spec, train, dev, cfg, start_from=artifact)
```

Decision rule: draft first. If `screen()` says meaningful headroom remains AND the
extra accuracy is worth roughly 10× the spend, run `optimize()` — full GEPA at full
budget still wins most tasks on absolute accuracy (it beat `draft` on 6 of 8 tasks
in the frozen pass; `draft` recovered 60–91% of its gain and won the other 2).
Honest caveat on `start_from`: draft-then-evolve is an UNTESTED combination — our
measured results cover draft alone and optimize alone.

## What the benchmarks say (read this before choosing features)

We benchmark honestly, including against ourselves — full protocol and numbers in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md). The pre-registered confirmation pass
(8 public datasets, fresh seeds, light budget, gpt-4.1-mini) found:

- **Vanilla GEPA over melvil's rendered prompt is the strongest configuration**
  (mean test accuracy 0.781 vs 0.755 for the full classification layer and 0.762 for
  MIPROv2; seed prompt 0.703). At light budgets, prefer `features=mv.Features.none()`.
- The classification layer's per-component updates trade whole-prompt coverage for
  structure; at ~6–10 accepted proposals per run that trade loses, especially on hard
  small-taxonomy tasks. Whether it wins at medium/heavy budgets is an open question.
- Hard-example mining, behind its (strict, quarantined) accept gate, kept exemplars in
  0/24 confirmation runs — treat it as a safety-gated no-op at light budgets.

## What the classification layer does

Each feature is independently toggleable via `mv.Features` (**all off by default** — the default is the benchmark-strongest vanilla-GEPA configuration; enable the layer with `mv.Features.all()` or per-flag;
`Features.none()` is an explicit alias for the default):

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
- **Offline testing** — `melvil.testing` ships fake LMs with a real
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
