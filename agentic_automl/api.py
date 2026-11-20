"""Local web interface; the existing comparison functions own all ML logic."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from math import isclose
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentic_automl.agents.experiment_analyst import ColumnExclusion, analyze_experiment
from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.classification.random_search import SEARCH_SPACES as CLASSIFIERS
from agentic_automl.data_review import apply_changes, review_dataset
from agentic_automl.history import (
    load_analyses,
    load_history,
    save_analysis,
    save_experiment,
    save_rejection,
)
from agentic_automl.regression.comparison import compare_regression_models
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
    except (OSError, ValueError, sqlite3.Error) as error:
        raise HTTPException(
            500, "Could not read the experiment history database."
        ) from error


@app.get("/api/experiments/{experiment_id}")
def experiment_details(experiment_id: str) -> dict:
    for experiment in experiment_history():
        if experiment["id"] == experiment_id:
            return experiment
    raise HTTPException(404, "Experiment not found.")


@app.post("/api/experiments/{experiment_id}/analysis")
def experiment_analysis(experiment_id: str) -> dict:
    experiment = experiment_details(experiment_id)
    try:
        analysis = analyze_experiment(experiment)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(
            502, "The analyst returned an invalid response. Try again."
        ) from error
    report = {
        **analysis,
        "id": str(uuid4()),
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    }
    try:
        save_analysis(report)
    except (OSError, ValueError) as error:
        raise HTTPException(
            500, "Analysis finished, but saving the report failed."
        ) from error
    return report


@app.get("/api/experiments/{experiment_id}/analyses")
def experiment_analyses(experiment_id: str) -> list[dict]:
    experiment_details(experiment_id)
    try:
        return load_analyses(experiment_id)
    except (OSError, ValueError) as error:
        raise HTTPException(500, "Could not read saved analyst reports.") from error


@app.post("/api/experiments/{experiment_id}/analyses/{analysis_id}/reject")
def reject_action(experiment_id: str, analysis_id: str) -> dict:
    if not any(
        report["id"] == analysis_id for report in experiment_analyses(experiment_id)
    ):
        raise HTTPException(404, "Analyst report not found.")
    try:
        save_rejection(
            {
                "analysis_id": analysis_id,
                "experiment_id": experiment_id,
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (OSError, ValueError) as error:
        raise HTTPException(500, "Could not save the rejection.") from error
    return {"rejected": True}


def read_upload(file: UploadFile, task: str, target_column: str) -> pd.DataFrame:
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV must be 25 MB or smaller.")
    data = pd.read_csv(
        BytesIO(content),
        dtype={target_column: str} if task == "classification" else None,
    )
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if data.empty or data.shape[1] < 2:
        raise ValueError("At least one feature column and one data row are required.")
    data.attrs["sha256"] = sha256(content).hexdigest()
    return data


@app.post("/api/dataset-review")
def dataset_review(
    task: Literal["classification", "regression"] = Form(...),
    target_column: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    try:
        return review_dataset(read_upload(file, task, target_column), target_column)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/experiments")
def run_experiment(
    task: Literal["classification", "regression"] = Form(...),
    target_column: str = Form(...),
    model_names: list[str] = Form(...),
    positive_class: str = Form("1"),
    apply_suggestions: bool = Form(False),
    file: UploadFile = File(...),
) -> dict:
    """Run one experiment, returning CV means and only the winner's test scores."""
    try:
        data = read_upload(file, task, target_column)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise HTTPException(400, str(error)) from error
    return execute_experiment(
        data,
        task,
        target_column,
        model_names,
        positive_class,
        apply_suggestions,
        file.filename or "dataset.csv",
    )


@app.post("/api/experiments/{experiment_id}/follow-up")
def approved_follow_up(
    experiment_id: str,
    analysis_id: str = Form(...),
    approved: bool = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if not approved:
        raise HTTPException(400, "Explicit approval is required to run a follow-up.")
    parent = experiment_details(experiment_id)
    reports = experiment_analyses(experiment_id)
    report = next((item for item in reports if item["id"] == analysis_id), None)
    if report is None or report.get("rejected") or not report.get("proposed_action"):
        raise HTTPException(
            400, "Choose a saved report with a proposed column exclusion."
        )
    if not parent.get("dataset_sha256"):
        raise HTTPException(
            400,
            "This older experiment has no dataset fingerprint. Create a new experiment first.",
        )
    try:
        action = ColumnExclusion.model_validate(report["proposed_action"])
        columns = {
            column["name"]
            for column in parent.get("training_profile", {}).get("columns", [])
        }
        if (
            action.column == parent["target_column"]
            or action.column not in columns
            or len(columns) < 2
        ):
            raise ValueError(
                "The suggested column is not an eligible training feature."
            )
        data = read_upload(file, parent["task"], parent["target_column"])
        if data.attrs["sha256"] != parent["dataset_sha256"]:
            raise ValueError(
                "Re-upload the exact same CSV used for the parent experiment."
            )
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise HTTPException(400, str(error)) from error
    excluded = [*parent.get("excluded_columns", []), action.column]
    return execute_experiment(
        data,
        parent["task"],
        parent["target_column"],
        parent["model_names"],
        parent["positive_class"],
        parent["dataset_review"]["accepted"],
        parent["dataset"],
        excluded_columns=excluded,
        parent=parent,
        report=report,
    )


def execute_experiment(
    data: pd.DataFrame,
    task: str,
    target_column: str,
    model_names: list[str],
    positive_class: str | None,
    apply_suggestions: bool,
    dataset: str,
    excluded_columns: list[str] | None = None,
    parent: dict | None = None,
    report: dict | None = None,
) -> dict:
    """Shared execution for initial experiments and approved CV-only follow-ups."""
    evaluate_test = parent is None
    excluded_columns = excluded_columns or []
    try:
        review = review_dataset(data, target_column)
        applied = review["proposed_changes"] if apply_suggestions else []
        prepared = apply_changes(data, applied)
        features, target = prepared.drop(columns=target_column), prepared[target_column]
        features = features.drop(columns=excluded_columns)
        if features.shape[1] == 0:
            raise ValueError("At least one feature must remain.")
        if task == "classification":
            features, target = classification_samples(features, target)
            table = compare_classification_models(
                features,
                target,
                positive_class=positive_class,
                model_names=model_names,
                evaluate_test=evaluate_test,
            )
        else:
            features, target = regression_samples(features, target)
            table = compare_regression_models(
                features, target, model_names=model_names, evaluate_test=evaluate_test
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
        "dataset": dataset,
        "dataset_sha256": data.attrs["sha256"],
        "excluded_columns": excluded_columns,
        "evaluation": "cv_and_test" if evaluate_test else "cv_only",
        "task": task,
        "target_column": target_column,
        "model_names": model_names,
        "positive_class": positive_class if task == "classification" else None,
        "sample_rows": len(features),
        "selected_model": winner,
        "selection_metric": table.attrs["selection_metric"],
        "rejected_columns": table.attrs["rejected_columns"],
        "training_profile": table.attrs["training_profile"],
        "dataset_review": {
            **review,
            "accepted": apply_suggestions,
            "applied_changes": applied,
            "prepared_rows": len(prepared),
        },
        "cv_results": json.loads(averages.reset_index().to_json(orient="records")),
        "test_results": json.loads(test.reset_index().to_json(orient="records"))
        if evaluate_test
        else [],
    }
    if parent is not None:
        metric = "f1" if task == "classification" else "mse"
        old_score = next(
            row[metric]
            for row in parent["cv_results"]
            if row["model"] == parent["selected_model"]
        )
        new_score = next(
            row[metric] for row in result["cv_results"] if row["model"] == winner
        )
        result.update(
            parent_id=parent["id"],
            analysis_id=report["id"],
            approved_action=report["proposed_action"],
            cv_comparison={
                "metric": metric,
                "parent": old_score,
                "follow_up": new_score,
                "improved": not isclose(
                    new_score, old_score, rel_tol=1e-9, abs_tol=1e-12
                )
                and (
                    new_score > old_score
                    if task == "classification"
                    else new_score < old_score
                ),
            },
        )
    try:
        save_experiment(result)
    except (OSError, ValueError, sqlite3.Error) as error:
        raise HTTPException(
            500, "Training finished, but saving experiment history failed."
        ) from error
    return result
