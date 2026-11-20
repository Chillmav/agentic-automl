import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agentic_automl.api import app

client = TestClient(app)


def parent_and_report(monkeypatch, task="classification", column="constant"):
    csv = pd.DataFrame(
        {
            "signal": [0] * 50 + [1] * 50,
            "constant": [1] * 100,
            "target": [0] * 50 + [1] * 50,
        }
    ).to_csv(index=False)
    model = "logistic_regression" if task == "classification" else "linear_regression"
    response = client.post(
        "/api/experiments",
        data={"task": task, "target_column": "target", "model_names": [model]},
        files={"file": ("data.csv", csv)},
    )
    assert response.status_code == 200, response.text
    parent = response.json()
    report = {
        "summary": "Consider removing the constant feature.",
        "findings": ["It has one distinct value."],
        "next_steps": ["Compare without that feature using CV."],
        "proposed_action": {
            "kind": "exclude_column",
            "column": column,
            "reason": "Constant in training.",
        },
    }
    monkeypatch.setattr(
        "agentic_automl.agents.experiment_analyst.review_with_ollama",
        lambda context: json.dumps(report),
    )
    response = client.post(f"/api/experiments/{parent['id']}/analysis")
    assert response.status_code == 200, response.text
    return parent, response.json(), csv


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_approved_exclusion_preserves_parent_and_only_evaluates_cv(
    monkeypatch, task
) -> None:
    parent, report, csv = parent_and_report(monkeypatch, task)
    response = client.post(
        f"/api/experiments/{parent['id']}/follow-up",
        data={"analysis_id": report["id"], "approved": "true"},
        files={"file": ("data.csv", csv)},
    )
    assert response.status_code == 200, response.text
    child = response.json()
    assert child["id"] != parent["id"]
    assert child["parent_id"] == parent["id"]
    assert child["analysis_id"] == report["id"]
    assert child["test_results"] == []
    assert child["evaluation"] == "cv_only"
    assert child["excluded_columns"] == ["constant"]
    assert child["dataset_sha256"] == parent["dataset_sha256"]
    assert child["model_names"] == parent["model_names"]
    assert child["training_profile"]["rows"] == parent["training_profile"]["rows"]
    assert [column["name"] for column in child["training_profile"]["columns"]] == [
        "signal"
    ]
    assert child["cv_comparison"]["parent"] == pytest.approx(
        child["cv_comparison"]["follow_up"], abs=1e-12
    )
    assert child["cv_comparison"]["improved"] is False
    assert client.get(f"/api/experiments/{parent['id']}").json() == parent


@pytest.mark.parametrize(
    "problem", ["no_approval", "changed_csv", "unknown_report", "rejected"]
)
def test_invalid_follow_up_never_runs_training(monkeypatch, problem) -> None:
    parent, report, csv = parent_and_report(monkeypatch)
    if problem == "rejected":
        response = client.post(
            f"/api/experiments/{parent['id']}/analyses/{report['id']}/reject"
        )
        assert response.status_code == 200
        assert (
            client.get(f"/api/experiments/{parent['id']}/analyses").json()[0][
                "rejected"
            ]
            is True
        )

    def should_not_train(*args, **kwargs):
        pytest.fail("An invalid action must be rejected before any training.")

    monkeypatch.setattr(
        "agentic_automl.api.compare_classification_models", should_not_train
    )
    response = client.post(
        f"/api/experiments/{parent['id']}/follow-up",
        data={
            "analysis_id": "missing" if problem == "unknown_report" else report["id"],
            "approved": "false" if problem == "no_approval" else "true",
        },
        files={
            "file": ("data.csv", csv + "1,1,1\n" if problem == "changed_csv" else csv)
        },
    )
    assert response.status_code == 400
    assert len(client.get("/api/experiments").json()) == 1
