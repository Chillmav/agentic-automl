import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.profiling import training_profile
from agentic_automl.regression.comparison import compare_regression_models


def test_profile_reports_missing_values_and_target_balance() -> None:
    features = pd.DataFrame(
        {"age": [10.0, None, 30.0, 40.0], "group": ["a", "b", "a", None]}
    )
    profile = training_profile(
        features, pd.Series(["yes", "yes", "yes", "no"]), "classification"
    )
    assert profile["target"]["class_counts"] == {"yes": 3, "no": 1}
    assert profile["columns"][0]["missing_percent"] == 25
    assert profile["columns"][0]["numeric"]["median"] == 30
    assert profile["columns"][1]["distinct"] == 2


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_pipeline_profile_excludes_test_rows(task) -> None:
    target = pd.Series(
        np.arange(100) % 2 if task == "classification" else np.arange(100) * 2
    )
    features = pd.DataFrame({"signal": target, "test_only_extreme": np.zeros(100)})
    train, test = train_test_split(
        features.index,
        test_size=0.2,
        random_state=42,
        stratify=target if task == "classification" else None,
    )
    features.loc[test, "test_only_extreme"] = 1_000_000
    if task == "classification":
        table = compare_classification_models(
            features, target, positive_class=1, model_names=["logistic_regression"]
        )
    else:
        table = compare_regression_models(
            features, target, model_names=["linear_regression"]
        )
    profile = table.attrs["training_profile"]
    assert profile["rows"] == len(train) == 80
    assert profile["columns"][1]["numeric"]["max"] == 0
    if task == "regression":
        assert profile["target"]["mean"] == pytest.approx(target.loc[train].mean())


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cv_only_mode_never_predicts_on_outer_test_rows(monkeypatch, task) -> None:
    target = pd.Series([0] * 50 + [1] * 50)
    features = pd.DataFrame({"signal": target})
    train, test = train_test_split(
        features.index,
        test_size=0.2,
        random_state=42,
        stratify=target if task == "classification" else None,
    )
    original_predict = Pipeline.predict
    seen = []

    def predict(pipeline, frame, **kwargs):
        assert set(frame.index).issubset(set(train))
        assert not set(frame.index).intersection(test)
        seen.append(frame.index)
        return original_predict(pipeline, frame, **kwargs)

    monkeypatch.setattr(Pipeline, "predict", predict)
    if task == "classification":
        table = compare_classification_models(
            features,
            target,
            positive_class=1,
            model_names=["logistic_regression"],
            evaluate_test=False,
        )
    else:
        table = compare_regression_models(
            features, target, model_names=["linear_regression"], evaluate_test=False
        )
    assert len(seen) == 3
    assert table.filter(regex=r"^test_").shape[1] == 0
