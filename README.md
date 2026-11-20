# Agentic AutoML

Tabular binary classification and regression experiments using scikit-learn.

## Run the local web interface

From the project directory:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn agentic_automl.api:app --reload
```

Open **http://127.0.0.1:8000** to see experiment history. Click **New experiment**
to choose a CSV, enter its target column, select models, and click **Review dataset**.
For classification, enter the exact positive label (for example,
`1` or `yes`). Uploads must be CSV text, even if the filename ends in `.xls`.
The upload limit is 25 MB; at most 10,000 rows are sampled with seed 42.

The page shows three-fold CV averages and a separate table for the winner's final
test scores. Classification selects by F1; regression selects by lowest MSE.
Baselines are always included. Parameter search is disabled in this interface.

Click a dataset filename in History (or **View experiment details** after a run)
to open `/experiments/{id}`. This page shows the saved configuration, all CV
averages, the winner's test scores and rejected columns. It does not retrain models.
Rejected-column counts come from the winning model's final training split;
CV folds may reject different columns. Older records show **Not recorded** for
this information, and baseline winners do not use feature preprocessing.

Use independent observations: the web interface uses a random 80/20 train/test
split. It does not expose the chronological split available in `main_regression.py`.
Remove ID columns and target-leaking features before upload. Existing categorical
preprocessing still rejects columns with more than 20 distinct training values.

## Dataset review

Before training, the app shows column types, missing/distinct counts, proposed
operations and notes about duplicates, categorical cardinality and random splits.
Choose **Accept all and train**, **Use original data and train**, or **Cancel review**
to edit your inputs. Suggestions are all-or-nothing in this first version.

The current proposals exclude rows with missing targets and replace valid,
consistent ISO date features (such as `2024-03-15`) with numeric year/month/day/weekday
columns. Timestamps are interpreted in UTC and time of day is discarded. Ambiguous
or mixed formats are not guessed, and existing column names are never overwritten.
Date expansion does not add chronological validation. Declining changes can leave
data problems that prevent training.

The source file is unchanged. Normal fold-local imputation, scaling, category
encoding and high-cardinality rejection still apply. The review, decision and
applied operations are saved with the experiment and shown on its details page.
This step uses ordinary Python rules. The optional experiment analyst below runs
separately, after training, only when requested.

## Local experiment analyst (LangGraph)

Open an experiment's details and click **Analyze experiment**. A small LangGraph
workflow runs `prepare_context -> recommend -> END`. It sends saved training
profiles and CV scores to a local Ollama model, and displays a summary,
findings and up to three suggested next steps. Test scores are excluded from the
model's input. It has no raw CSV rows or fitted
model, so column-specific conclusions are limited to the recorded metadata.

The workflow makes one model call per click, with no retries or automatic actions.
Every report is saved with a date and model name in `results/analyses.jsonl`.
The **Saved report** selector restores all versions without another model call.
The analyst may return one typed column-exclusion suggestion with a reason, or
no action when the available evidence does not support one. Model advice can be wrong.

New experiments also store a profile computed from the outer training split:
feature types, missing percentages, distinct counts, numeric min/max/mean/median,
and target class counts or numeric target statistics. Profiles appear on the
details page. Older experiments remain readable but have no recorded profile.

## Approving a follow-up

When a saved report proposes a column exclusion, re-upload the **exact original
CSV** and click **Approve exclusion and run CV**, or click **Reject suggestion**.
Rejections are saved in `results/analysis_decisions.jsonl` and remain after reload.
No dataset is stored; a SHA-256 fingerprint verifies the re-uploaded bytes.

An approved follow-up keeps the parent's task, chosen models, preparation and
seeded sampling/splits, then excludes the approved column. It saves a new
experiment linked to its parent and report, and shows the before/after winner CV
scores. It does not predict on or score the final test set again. Follow-up history
rows are marked **CV**, while initial runs show **Test** scores. Further exclusions
can be proposed one at a time, with cumulative excluded columns preserved.

This first action is limited to excluding an existing feature while keeping at
least one feature. Arbitrary model-generated code is never executed. Older runs
without a file fingerprint cannot launch verified follow-ups: create a fresh
initial experiment first. Source uploads and previous results stay unchanged.

Install [Ollama for Linux](https://docs.ollama.com/linux) outside the virtual
environment, then download and test the small default model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

[Qwen2.5 3B](https://ollama.com/library/qwen2.5:3b) is a 1.9 GB model download;
Ollama and its runtime libraries require additional storage. If the server is not
running, start `ollama serve` in a separate terminal. The backend connects to
`http://127.0.0.1:11434/api/chat`. No API key is needed for this local setup.

After installing the updated Python requirements, start the app as usual. To use
a different already-downloaded Ollama model:

```bash
OLLAMA_MODEL=qwen2.5:3b uvicorn agentic_automl.api:app --reload
```

`agents/experiment_analyst.py` owns the graph and response validation;
`agents/ollama.py` owns the model request and prompt. A different provider can
replace this callable later. The request has a 90-second timeout, a 700-token
generation cap and a 20,000-character input-summary limit. The graph API follows
the [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/graph-api).

## How the web layer works

1. `frontend/index.html` loads React, ReactDOM and Babel, and provides a root element.
2. `frontend/app.js` contains `App` for new experiments, `ResultsTable` for scores,
   `History` for past runs, and `ExperimentDetails` for a saved run. The HTML-like
   syntax is JSX. Ordinary links switch between pages.
3. `agentic_automl/api.py` validates the request and calls the existing ML pipeline.
4. `agentic_automl/history.py` routes experiment storage to `database.py` (SQLite),
   and appends/reads analyst reports and decisions in JSONL.
5. `agentic_automl/data_review.py` proposes and applies the supported operations.
6. `agentic_automl/profiling.py` describes training data; `frontend/analyst.js`
   handles saved reports, approval and rejection.

FastAPI serves both the frontend and API, so startup stays the same. No Node.js,
npm or separate frontend server is needed. React and Babel are loaded from pinned
CDN URLs, so the browser needs internet access. Babel translates JSX in the browser;
this keeps the development setup small. For deployment, replace browser compilation
with a build step. See the [Babel standalone documentation](https://babeljs.io/docs/babel-standalone).

API documentation is at **http://127.0.0.1:8000/docs**.
`GET /api/models` lists choices; `POST /api/experiments` accepts the form upload;
`GET /api/experiments` returns saved runs, newest first.
`GET /api/experiments/{id}` returns one saved run, or 404 if it does not exist.
`POST /api/dataset-review` inspects an upload without training or saving history.
Training accepts `apply_suggestions=true`; it defaults to false for API callers.
`POST /api/experiments/{id}/analysis` runs the local analyst on a saved experiment.
`GET /api/experiments/{id}/analyses` lists every saved report.
`POST /api/experiments/{id}/analyses/{report_id}/reject` records a rejection.
`POST /api/experiments/{id}/follow-up` requires `analysis_id`, `approved=true`
and the original CSV upload.

## Experiment storage (SQLite, first increment)

Experiments run synchronously: the browser waits for the response. Successful web
runs are saved to **`results/history.sqlite3`**, using Python's built-in `sqlite3`, and
remain available after browser or server restarts. Each entry stores a UTC date,
dataset filename, task, target, chosen models, positive class, row count, CV averages
and winner's test scores (initial runs only). New records also store rejected-column
information, training profiles and file fingerprints; follow-ups include their
parent, approved action and CV comparison.
The history page displays dates in your browser's local
time. Older experiments run before history was added are not available.

The `experiments` table has ID, completion date, dataset, task, parent ID and a JSON
payload containing the full configuration, scores and profile. Connections use
short transactions and are closed after each operation. Parent relationships are
checked by SQLite. Saving an existing ID never replaces its results.

To import existing experiments once, run from the project directory before using
the updated app:

```bash
python -m agentic_automl.migrate_history
```

The command imports `results/experiments.jsonl` into SQLite, preserving IDs and
dates so existing history links and report associations continue to work. Repeating
it skips IDs already in the database. An invalid record rolls back the entire
import and reports the error; the source JSONL is never changed or removed.
New experiments only go to SQLite. Once migrated, the JSONL file is a backup,
and changes to it do not affect the app unless you explicitly import again.

This first increment migrates **experiments only**. Reports remain in
`results/analyses.jsonl` and decisions in `results/analysis_decisions.jsonl` until
the next increment. Preserve all three storage locations when backing up the app.

The database and result files are ignored by Git. The app is intended for local use;
all visitors to that server share the same history. CLI runs are not recorded.
Uploaded CSVs are read in memory; neither CSV contents nor
trained models are saved. This is a local development interface without authentication.
The existing command-line experiment scripts continue to work.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```
