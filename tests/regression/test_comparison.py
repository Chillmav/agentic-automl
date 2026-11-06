from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from agentic_automl.regression.comparison import compare_regression_models
from agentic_automl.regression.data import load_regression_csv
from agentic_automl.regression.models.linear_regression import train_linear_regression


def test_csv_to_regression_comparison_fits_training_rows_only(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"size": range(100), "price": [2 * x + 1 for x in range(100)]}).to_csv(
        path, index=False
    )
    features, target = load_regression_csv(path, "price")
    train_x, _, train_y, _ = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    with patch(
        "agentic_automl.regression.comparison.train_linear_regression",
        wraps=train_linear_regression,
    ) as train:
        table = compare_regression_models(features, target)
    train.assert_called_once()
    pd.testing.assert_frame_equal(train.call_args.args[0], train_x)
    pd.testing.assert_series_equal(train.call_args.args[1], train_y)
    assert table.columns.tolist() == ["mse", "r2"]
    assert table.loc["linear_regression", "mse"] == pytest.approx(0, abs=1e-10)
    assert table.loc["linear_regression", "r2"] == pytest.approx(1)
    assert table.loc["mean_baseline", "mse"] > table.loc["linear_regression", "mse"]
    pd.testing.assert_frame_equal(table, compare_regression_models(features, target))


def test_too_few_test_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="two test rows"):
        compare_regression_models(pd.DataFrame({"size": range(5)}), pd.Series(range(5)))


def test_ordered_split_trains_only_on_earlier_rows() -> None:
    features, target = pd.DataFrame({"size": range(100)}), pd.Series(range(100))
    with patch(
        "agentic_automl.regression.comparison.train_linear_regression",
        wraps=train_linear_regression,
    ) as train:
        compare_regression_models(features, target, shuffle=False)
    assert train.call_args.args[0].index.tolist() == list(range(80))
