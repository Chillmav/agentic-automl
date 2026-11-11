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
to choose a CSV, enter its target column, select
models, and run. For classification, enter the exact positive label (for example,
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

## How the web layer works

1. `frontend/index.html` loads React, ReactDOM and Babel, and provides a root element.
2. `frontend/app.js` contains `App` for new experiments, `ResultsTable` for scores,
   `History` for past runs, and `ExperimentDetails` for a saved run. The HTML-like
   syntax is JSX. Ordinary links switch between pages.
3. `agentic_automl/api.py` validates the request and calls the existing ML pipeline.
4. `agentic_automl/history.py` appends and reads saved results.

FastAPI serves both the frontend and API, so startup stays the same. No Node.js,
npm or separate frontend server is needed. React and Babel are loaded from pinned
CDN URLs, so the browser needs internet access. Babel translates JSX in the browser;
this keeps the development setup small. For deployment, replace browser compilation
with a build step. See the [Babel standalone documentation](https://babeljs.io/docs/babel-standalone).

API documentation is at **http://127.0.0.1:8000/docs**.
`GET /api/models` lists choices; `POST /api/experiments` accepts the form upload;
`GET /api/experiments` returns saved runs, newest first.
`GET /api/experiments/{id}` returns one saved run, or 404 if it does not exist.

Experiments run synchronously: the browser waits for the response. Successful web
runs are saved to **`results/experiments.jsonl`**, one JSON object per line, and
remain available after browser or server restarts. Each entry stores a UTC date,
dataset filename, task, target, chosen models, positive class, row count, CV averages
and winner's test scores. New records also store rejected-column information.
The history page displays dates in your browser's local
time. Older experiments run before history was added are not available.

The file is ignored by Git. This storage is for a single local server process;
all visitors to that server share the same history. CLI runs are not recorded.
Uploaded CSVs use temporary files removed after the run; neither CSV contents nor
trained models are saved. This is a local development interface without authentication.
The existing command-line experiment scripts continue to work.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```
