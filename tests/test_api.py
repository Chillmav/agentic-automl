import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agentic_automl.api import app
from agentic_automl.history import save_experiment

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agentic_automl.history.HISTORY_FILE", tmp_path / "history.jsonl"
    )


def test_frontend_and_model_choices() -> None:
    assert "AutoML experiments" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/new").status_code == 200
    choices = client.get("/api/models").json()
    assert "logistic_regression" in choices["classification"]
    assert "ridge" in choices["regression"]


@pytest.mark.parametrize(
    "task,model,labels,metrics",
    [
        (
            "classification",
            "logistic_regression",
            ["no"] * 50 + ["yes"] * 50,
            {"acc", "recall", "precision", "f1"},
        ),
        ("regression", "linear_regression", list(range(100)), {"mse", "r2"}),
    ],
)
def test_uploaded_csv_runs_selected_model(task, model, labels, metrics) -> None:
    csv = pd.DataFrame(
        {
            "signal": range(100),
            "identifier": [f"row{i}" for i in range(100)],
            "target": labels,
        }
    ).to_csv(index=False)
    response = client.post(
        "/api/experiments",
        data={
            "task": task,
            "target_column": "target",
            "model_names": [model],
            "positive_class": "yes",
        },
        files={"file": ("data.csv", csv, "text/csv")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["sample_rows"] == 100
    assert result["dataset"] == "data.csv"
    assert result["task"] == task
    assert result["target_column"] == "target"
    assert result["model_names"] == [model]
    assert result["completed_at"].endswith("+00:00")
    assert client.get("/api/experiments").json() == [result]
    assert result["rejected_columns"] == {"identifier": 80}
    assert client.get(f"/api/experiments/{result['id']}").json() == result
    assert client.get(f"/experiments/{result['id']}").status_code == 200
    assert {row["model"] for row in result["cv_results"]} == (
        {model, "mean_baseline"}
        if task == "regression"
        else {model, "guessing_baseline", "majority_baseline"}
    )
    assert len(result["test_results"]) == 1
    assert result["test_results"][0]["model"] == result["selected_model"]
    for row in result["cv_results"] + result["test_results"]:
        assert set(row) == metrics | {"model"}
        assert all(row[metric] is not None for metric in metrics)


@pytest.mark.parametrize(
    "csv,target,models,message",
    [
        ("x,y\n1,2\n", "missing", ["ridge"], "not found"),
        ("x,y\n1,no\n2,yes\n", "y", ["ridge"], "finite numbers"),
        ("", "y", ["ridge"], "No columns"),
        ("x,y\n1,2\n2,3\n", "y", ["unknown"], "Unknown regressors"),
    ],
)
def test_invalid_upload_returns_readable_error(csv, target, models, message) -> None:
    response = client.post(
        "/api/experiments",
        data={"task": "regression", "target_column": target, "model_names": models},
        files={"file": ("data.csv", csv)},
    )
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert client.get("/api/experiments").json() == []


def test_upload_size_limit(monkeypatch) -> None:
    monkeypatch.setattr("agentic_automl.api.MAX_UPLOAD_BYTES", 5)
    response = client.post(
        "/api/experiments",
        data={"task": "regression", "target_column": "y", "model_names": ["ridge"]},
        files={"file": ("data.csv", "x,y\n1,2\n")},
    )
    assert response.status_code == 413


def test_empty_history() -> None:
    assert client.get("/api/experiments").json() == []


@pytest.mark.parametrize("endpoint", ["/api/experiments", "/api/experiments/some-id"])
def test_unreadable_history_returns_error(tmp_path, endpoint) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text("broken JSON\n", encoding="utf-8")
    response = client.get(endpoint)
    assert response.status_code == 500
    assert "Could not read" in response.json()["detail"]
    assert path.read_text(encoding="utf-8") == "broken JSON\n"


def test_failed_save_is_reported(monkeypatch) -> None:
    def fail_save(record: dict) -> None:
        raise OSError("Disk is full")

    monkeypatch.setattr("agentic_automl.api.save_experiment", fail_save)
    csv = pd.DataFrame({"x": range(50), "y": range(50)}).to_csv(index=False)
    response = client.post(
        "/api/experiments",
        data={"task": "regression", "target_column": "y", "model_names": ["ridge"]},
        files={"file": ("data.csv", csv)},
    )
    assert response.status_code == 500
    assert "Training finished, but saving" in response.json()["detail"]
    assert client.get("/api/experiments").json() == []


def test_unknown_experiment_returns_404() -> None:
    response = client.get("/api/experiments/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found."


def test_details_preserve_old_records_without_rejected_columns() -> None:
    record = {"id": "old", "dataset": "old.csv", "selected_model": "ridge"}
    save_experiment(record)
    save_experiment({"id": "new", "dataset": "new.csv"})
    assert client.get("/api/experiments/old").json() == record
    assert "rejected_columns" not in client.get("/api/experiments/old").json()
