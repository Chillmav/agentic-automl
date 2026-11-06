from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agentic_automl.comparison import compare_models
from agentic_automl.data import load_classification_csv, stratified_split
from agentic_automl.model import train_logistic_regression


def test_csv_to_comparison_beats_guessing_on_clear_signal(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"signal": [0] * 80 + [1] * 20, "target": ["no"] * 80 + ["yes"] * 20}
    ).to_csv(path, index=False)
    features, target = load_classification_csv(path, "target")
    results = compare_models(features, target)
    assert results["logistic_regression"]["accuracy"] == 1.0
    assert results["logistic_regression"]["balanced_accuracy"] == 1.0
    assert results["guessing_baseline"]["balanced_accuracy"] < 1.0
    assert results == compare_models(features, target)
    for scores in results.values():
        assert all(0 <= score <= 1 for score in scores.values())


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
            features, target, categorical_columns=["code"], test_size=0.3
        )
    train.assert_called_once()
    pd.testing.assert_frame_equal(train.call_args.args[0], expected_train_x)
    pd.testing.assert_series_equal(train.call_args.args[1], expected_train_y)
    assert train.call_args.kwargs["categorical_columns"] == ["code"]
    assert results["logistic_regression"]["accuracy"] == 1.0
