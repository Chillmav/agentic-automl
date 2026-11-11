"""Local web interface; the existing comparison functions own all ML logic."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.classification.data import load_classification_csv
from agentic_automl.classification.random_search import SEARCH_SPACES as CLASSIFIERS
from agentic_automl.history import load_history, save_experiment
from agentic_automl.regression.comparison import compare_regression_models
from agentic_automl.regression.data import load_regression_csv
from agentic_automl.regression.random_search import SEARCH_SPACES as REGRESSORS
from main_classification import limit_samples as classification_samples
from main_regression import limit_samples as regression_samples

app = FastAPI(title="AutoML local experiments")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
@app.get("/history", include_in_schema=False)
@app.get("/new", include_in_schema=False)
@app.get("/experiments/{experiment_id}", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/models")
def available_models() -> dict[str, list[str]]:
    return {"classification": list(CLASSIFIERS), "regression": list(REGRESSORS)}


@app.get("/api/experiments")
def experiment_history() -> list[dict]:
    try:
        return load_history()
    except (OSError, ValueError) as error:
        raise HTTPException(
            500, "Could not read the experiment history file."
        ) from error


@app.get("/api/experiments/{experiment_id}")
def experiment_details(experiment_id: str) -> dict:
    for experiment in experiment_history():
        if experiment["id"] == experiment_id:
            return experiment
    raise HTTPException(404, "Experiment not found.")


@app.post("/api/experiments")
def run_experiment(
    task: Literal["classification", "regression"] = Form(...),
    target_column: str = Form(...),
    model_names: list[str] = Form(...),
    positive_class: str = Form("1"),
    file: UploadFile = File(...),
) -> dict:
    """Run one experiment, returning CV means and only the winner's test scores."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV must be 25 MB or smaller.")
    try:
        # A fixed temporary filename avoids using the user's filename as a path.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_bytes(content)
            if task == "classification":
                features, target = load_classification_csv(path, target_column)
                features, target = classification_samples(features, target)
                table = compare_classification_models(
                    features,
                    target,
                    positive_class=positive_class,
                    model_names=model_names,
                )
            else:
                features, target = load_regression_csv(path, target_column)
                features, target = regression_samples(features, target)
                table = compare_regression_models(
                    features,
                    target,
                    model_names=model_names,
                )
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise HTTPException(400, str(error)) from error

    winner = table.attrs["selected_model"]
    averages = table.filter(regex=r"^cv_.*_mean$").rename(
        columns=lambda name: name.removeprefix("cv_").removesuffix("_mean")
    )
    test = (
        table.filter(regex=r"^test_")
        .loc[[winner]]
        .rename(columns=lambda name: name.removeprefix("test_"))
    )
    # pandas converts numpy values to JSON numbers and nonfinite scores to null.
    result = {
        "id": str(uuid4()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": file.filename or "dataset.csv",
        "task": task,
        "target_column": target_column,
        "model_names": model_names,
        "positive_class": positive_class if task == "classification" else None,
        "sample_rows": len(features),
        "selected_model": winner,
        "selection_metric": table.attrs["selection_metric"],
        "rejected_columns": table.attrs["rejected_columns"],
        "cv_results": json.loads(averages.reset_index().to_json(orient="records")),
        "test_results": json.loads(test.reset_index().to_json(orient="records")),
    }
    try:
        save_experiment(result)
    except (OSError, ValueError) as error:
        raise HTTPException(
            500, "Training finished, but saving experiment history failed."
        ) from error
    return result
