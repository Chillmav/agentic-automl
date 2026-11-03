# Agentic AutoML

An experimental platform for budgeted tabular ML. The first step is a local,
reproducible binary classification experiment with **zero LLM/API cost**.

## Setup and run

Python 3.12 is the initial supported environment. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m agentic_automl data.csv \
  --target outcome --positive-class yes --output /tmp/automl-run-001
```

Use the exact positive label from your CSV (for example `1` or `yes`). Target
labels are read as strings. Missing target labels, nonbinary targets, and datasets
with fewer than five examples per class are rejected. Features may contain missing
values. Inputs must have independent rows; temporal/grouped data and identifier
or leakage columns must be handled by the researcher before running.

The experiment uses an 80/20 stratified development/holdout split and shared
3-fold stratified CV within development data. Numeric columns receive median
imputation and scaling; categorical columns receive mode imputation and one-hot
encoding with at most 32 categories per column. Preprocessing is fitted inside
each fold. A dummy classifier and logistic regression compete on mean CV ROC-AUC.
Boolean feature columns pass through as binary values.
Ties prefer the first candidate (dummy). Only the winner is fitted on all
development rows and evaluated on holdout. The exported pipeline retains this
development-data fit so its holdout result remains attributable to that artifact.

Each new output directory contains:

- `report.json`: CV fold scores, mean/std, winner, holdout ROC-AUC, average precision,
  balanced accuracy, F1, elapsed time, and zero LLM cost.
- `pipeline.joblib`: fitted preprocessing and classifier; probabilities in column 1
  refer to the explicitly chosen positive class. Load only trusted joblib files.
- `manifest.json`: dataset SHA-256, class mapping, feature schema, exact split/fold
  row positions, seed, pipeline parameters, Python and dependency versions.

Row indices are zero-based positions in the parsed CSV. The same file, pinned
environment, and seed (`42` by default) reproduce splits and scores; exact timings
and cross-platform bitwise equality are not guaranteed. Preserve the original CSV
and the source revision alongside artifacts for future replay. Existing output
directories are refused, never overwritten. Failed writes can leave a partial
directory; rerun using a new output path.

## Validation

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Next steps and cost

Next add random search and Optuna using this same evaluation policy, then a bounded
LangGraph planner. Future agent experiments should start with an estimated USD
0.05/run LLM budget, configurable downward or disabled; pricing/token accounting and
budget enforcement must be implemented before paid calls are enabled. This first
step makes no network calls and uses no paid services. CPU time and memory are not
yet hard-limited. Start with small datasets.

See [architecture decisions](docs/architecture.md) for scope and expansion boundaries.
