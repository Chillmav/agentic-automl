import copy
import json
from unittest.mock import Mock

import httpx
import pytest

from agentic_automl.agents.experiment_analyst import analyze_experiment
from agentic_automl.agents.ollama import review_with_ollama

REPORT = {
    "summary": "Ridge improved on the mean baseline during CV.",
    "findings": ["The dataset summary was not recorded."],
    "next_steps": ["Compare another regressor on the same CV folds."],
}


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_graph_calls_model_once_without_changing_record(task) -> None:
    record = {
        "id": "run",
        "task": task,
        "selected_model": "ridge",
        "cv_results": [],
        "extra_unrelated_field": "exclude this",
    }
    original = copy.deepcopy(record)
    reviewer = Mock(return_value=json.dumps(REPORT))
    assert analyze_experiment(record, reviewer) == {**REPORT, "proposed_action": None}
    reviewer.assert_called_once()
    context = json.loads(reviewer.call_args.args[0])
    assert context["task"] == task
    assert "dataset_review" not in context  # Older records remain usable.
    assert "extra_unrelated_field" not in context
    assert record == original


def test_graph_includes_saved_dataset_evidence() -> None:
    record = {
        "dataset_review": {"columns": [{"name": "age", "missing": 3}]},
        "rejected_columns": {"identifier": 80},
        "cv_results": [{"model": "ridge", "mse": 10}],
        "test_results": [{"model": "ridge", "mse": 12}],
    }
    reviewer = Mock(return_value=json.dumps(REPORT))
    analyze_experiment(record, reviewer)
    context = json.loads(reviewer.call_args.args[0])
    assert context == {
        key: value for key, value in record.items() if key != "test_results"
    }


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "{}",
        json.dumps({**REPORT, "next_steps": []}),
        json.dumps({**REPORT, "next_steps": ["step"] * 4}),
    ],
)
def test_graph_rejects_invalid_report_without_retry(response) -> None:
    reviewer = Mock(return_value=response)
    with pytest.raises(ValueError):
        analyze_experiment({"task": "regression"}, reviewer)
    reviewer.assert_called_once()


def test_oversized_input_is_rejected_before_model_call() -> None:
    reviewer = Mock()
    with pytest.raises(RuntimeError, match="20,000-character"):
        analyze_experiment({"dataset_review": {"warnings": ["x" * 20_000]}}, reviewer)
    reviewer.assert_not_called()


def test_ollama_uses_local_endpoint_and_bounded_request(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "my-local-model")
    post = Mock(
        return_value=httpx.Response(
            200,
            json={"message": {"content": json.dumps(REPORT)}},
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
        )
    )
    monkeypatch.setattr("agentic_automl.agents.ollama.httpx.post", post)
    assert json.loads(review_with_ollama('{"task":"regression"}')) == REPORT
    assert post.call_args.args[0] == "http://127.0.0.1:11434/api/chat"
    request = post.call_args.kwargs
    assert request["timeout"] == 90
    assert request["json"]["model"] == "my-local-model"
    assert request["json"]["stream"] is False
    assert request["json"]["options"]["num_predict"] == 700
    assert "final test set" in request["json"]["messages"][0]["content"]


def test_unavailable_ollama_has_actionable_error(monkeypatch) -> None:
    post = Mock(side_effect=httpx.ConnectError("offline"))
    monkeypatch.setattr("agentic_automl.agents.ollama.httpx.post", post)
    with pytest.raises(RuntimeError, match="ollama serve"):
        review_with_ollama("{}")
    post.assert_called_once()


@pytest.mark.parametrize("column", ["target", "unknown", "constant"])
def test_proposed_action_only_accepts_an_existing_training_feature(column) -> None:
    record = {
        "target_column": "target",
        "training_profile": {"columns": [{"name": "signal"}, {"name": "constant"}]},
        "test_results": [{"model": "example", "f1": 0.99}],
    }
    reviewer = Mock(
        return_value=json.dumps(
            {
                **REPORT,
                "proposed_action": {
                    "kind": "exclude_column",
                    "column": column,
                    "reason": "Constant in training data.",
                },
            }
        )
    )
    if column == "constant":
        assert (
            analyze_experiment(record, reviewer)["proposed_action"]["column"] == column
        )
    else:
        with pytest.raises(ValueError):
            analyze_experiment(record, reviewer)
    assert "test_results" not in json.loads(reviewer.call_args.args[0])
