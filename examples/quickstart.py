"""The README quickstart, end-to-end on a real public dataset (AG News).

Prepares a small CSV from HuggingFace, then runs the exact quickstart lines.
Cost: well under $1 with the default models (a "light" budget run).

    export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY + the openai/ model ids
    python examples/quickstart.py
"""

import csv
import logging
import pathlib

import labelsmith as ls

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --- prepare a small CSV (stand-in for "your labeled data") ------------------
csv_path = pathlib.Path("agnews_small.csv")
if not csv_path.exists():
    rows = ls.load_hf("fancyzhx/ag_news", split="train", limit=1200)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows((e.text, e.label) for e in rows)

# --- the quickstart (verbatim in README) -------------------------------------
examples = ls.load_csv("agnews_small.csv")             # text,label columns
train, dev = ls.train_dev_split(examples, dev_size=100, seed=0)
spec = ls.TaskSpec.from_examples("agnews-topics", examples)
cfg = ls.Config(task_model="openrouter/openai/gpt-4.1-mini",
                reflection_model="openrouter/openai/gpt-4.1", budget="light")
print(ls.estimate_optimize_cost(spec, train, dev, cfg))  # dry-run, before spending
artifact = ls.optimize(spec, train, dev, cfg, resume=True)
print(artifact.render())                               # deployable prompt string
artifact.save("agnews_topics.v1.json")

# --- evaluate on held-out test data + markdown report ------------------------
test = ls.load_hf("fancyzhx/ag_news", split="test", limit=300)
rep = ls.evaluate(artifact, test)
print()
print(ls.report(artifact))
print()
print(ls.report(rep))
