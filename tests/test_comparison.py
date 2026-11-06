from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_automl.comparison import compare_models
from agentic_automl.data import load_classification_csv, stratified_split
from agentic_automl.model import train_logistic_regression


def test_csv_to_comparison_beats_guessing_on_clear_signal(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"signal": [0] * 80 + [1] * 20, "target": ["no"] * 80 + ["yes"] * 20}
    ).to_csv(path, index=False)
    features, target = load_classification_csv(path, "target")
    results = compare_models(features, target, positive_class="yes")
    assert results.columns.tolist() == ["acc", "recall", "precision", "f1"]
    assert results.loc["logistic_regression"].eq(1.0).all()
    assert results.loc["guessing_baseline", "acc"] < 1.0
    assert results.loc["majority_baseline"].to_dict() == {
        "acc": 0.8,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
    }
    pd.testing.assert_frame_equal(
        results, compare_models(features, target, positive_class="yes")
    )
    assert ((results >= 0) & (results <= 1)).all().all()

    negative_as_positive = compare_models(features, target, positive_class="no")
    assert negative_as_positive.loc["majority_baseline", "recall"] == 1.0
    assert negative_as_positive.loc["majority_baseline", "precision"] == 0.8
    assert negative_as_positive.loc["majority_baseline", "f1"] == pytest.approx(8 / 9)


def test_comparison_passes_only_training_rows_and_category_overrides() -> None:
    features = pd.DataFrame({"code": [1, 2] * 50})
    target = pd.Series(["no", "yes"] * 50)
    expected_train_x, _, expected_train_y, _ = stratified_split(
        features, target, test_size=0.3
    )
    with patch(
        "agentic_automl.comparison.train_logistic_regression",
        wraps=train_logistic_regression,
    ) as train:
        results = compare_models(
            features,
            target,
            categorical_columns=["code"],
            test_size=0.3,
            positive_class="yes",
        )
    train.assert_called_once()
    pd.testing.assert_frame_equal(train.call_args.args[0], expected_train_x)
    pd.testing.assert_series_equal(train.call_args.args[1], expected_train_y)
    assert train.call_args.kwargs["categorical_columns"] == ["code"]
    assert results.loc["logistic_regression", "acc"] == 1.0


def test_unknown_positive_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Positive class.*missing"):
        compare_models(
            pd.DataFrame({"value": range(20)}),
            pd.Series([0, 1] * 10),
            positive_class="missing",
        )
