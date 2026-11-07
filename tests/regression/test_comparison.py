from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agentic_automl.regression.comparison import compare_regression_models
from agentic_automl.regression.data import load_regression_csv
from agentic_automl.regression.models.decision_tree import build_decision_tree
from agentic_automl.regression.models.gradient_boosting import build_gradient_boosting
from agentic_automl.regression.models.linear_regression import build_linear_regression
from agentic_automl.regression.models.random_forest import build_random_forest
from agentic_automl.regression.models.ridge import build_ridge
from agentic_automl.regression.validation import cross_validation_scores


@pytest.mark.parametrize(
    "trainer",
    [
        build_linear_regression,
        build_ridge,
        build_decision_tree,
        build_random_forest,
        build_gradient_boosting,
    ],
)
def test_csv_to_regression_comparison_uses_unfitted_builders(
    tmp_path: Path, trainer
) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"size": range(100), "price": [2 * x + 1 for x in range(100)]}).to_csv(
        path, index=False
    )
    features, target = load_regression_csv(path, "price")
    with patch(
        f"agentic_automl.regression.comparison.{trainer.__name__}",
        wraps=trainer,
    ) as train:
        table = compare_regression_models(features, target)
    train.assert_called_once_with(categorical_columns=None)
    assert table.columns.tolist() == [
        "cv_mse_mean",
        "cv_mse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "test_mse",
        "test_r2",
    ]
    assert table.index.tolist() == [
        "mean_baseline",
        "linear_regression",
        "ridge",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]
    assert table.attrs["selected_model"] == "linear_regression"
    assert table.loc["linear_regression", "test_mse"] == pytest.approx(0, abs=1e-10)
    assert table.loc["linear_regression", "test_r2"] == pytest.approx(1)
    assert (
        table.drop(index="linear_regression")[["test_mse", "test_r2"]]
        .isna()
        .all()
        .all()
    )
    assert (
        table.drop(index="mean_baseline")["cv_mse_mean"]
        < table.loc["mean_baseline", "cv_mse_mean"]
    ).all()
    pd.testing.assert_frame_equal(table, compare_regression_models(features, target))


def test_too_few_test_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="two test rows"):
        compare_regression_models(pd.DataFrame({"size": range(5)}), pd.Series(range(5)))


def test_ordered_split_trains_only_on_earlier_rows() -> None:
    features, target = pd.DataFrame({"size": range(100)}), pd.Series(range(100))
    fitted_rows = []
    original_fit = Pipeline.fit

    def record_fit(model, fit_features, fit_target, **kwargs):
        fitted_rows.append(fit_features.index.tolist())
        return original_fit(model, fit_features, fit_target, **kwargs)

    with patch.object(Pipeline, "fit", new=record_fit):
        compare_regression_models(features, target, shuffle=False)
    assert fitted_rows[-1] == list(range(80))


def test_cv_uses_shared_folds_without_final_test_rows() -> None:
    features, target = pd.DataFrame({"size": range(40)}), pd.Series(range(40))
    expected_x, _, expected_y, _ = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    with patch(
        "agentic_automl.regression.comparison.cross_validation_scores",
        wraps=cross_validation_scores,
    ) as validate:
        compare_regression_models(features, target)
    assert validate.call_count == 6
    folds = validate.call_args_list[0].args[3]
    for call in validate.call_args_list:
        pd.testing.assert_frame_equal(call.args[1], expected_x)
        pd.testing.assert_series_equal(call.args[2], expected_y)
        assert call.args[3] is folds


@pytest.mark.parametrize("winning_index", [0, 4])
def test_only_lowest_cv_mse_winner_predicts_test_rows(winning_index: int) -> None:
    features, target = pd.DataFrame({"size": range(40)}), pd.Series(range(40))
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    calls = []
    baseline_calls = []
    original_baseline_predict = DummyRegressor.predict
    fits = []
    original_pipeline_fit = Pipeline.fit
    original_baseline_fit = DummyRegressor.fit

    def fit_pipeline(model, fit_features, fit_target):
        fits.append((fit_features, fit_target))
        return original_pipeline_fit(model, fit_features, fit_target)

    def fit_baseline(model, fit_features, fit_target, **kwargs):
        if isinstance(fit_features, pd.DataFrame):
            fits.append((fit_features, fit_target))
        return original_baseline_fit(model, fit_features, fit_target, **kwargs)

    def predict_pipeline(model, test_features):
        calls.append(test_features)
        return test_y.to_numpy()

    def predict_baseline(model, prediction_features):
        # Gradient boosting also uses DummyRegressor internally during fitting.
        if isinstance(
            prediction_features, pd.DataFrame
        ) and prediction_features.index.equals(test_x.index):
            baseline_calls.append(prediction_features)
        return original_baseline_predict(model, prediction_features)

    scores = [
        {"cv_mse_mean": 0.1 if index == winning_index == 4 else 2.0}
        for index in range(6)
    ]
    with (
        patch(
            "agentic_automl.regression.comparison.cross_validation_scores",
            side_effect=scores,
        ),
        patch.object(Pipeline, "predict", new=predict_pipeline),
        patch.object(DummyRegressor, "predict", new=predict_baseline),
        patch.object(Pipeline, "fit", new=fit_pipeline),
        patch.object(DummyRegressor, "fit", new=fit_baseline),
    ):
        table = compare_regression_models(features, target)
    assert len(calls) + len(baseline_calls) == 1
    assert len(fits) == 1
    pd.testing.assert_frame_equal(fits[0][0], train_x)
    pd.testing.assert_series_equal(fits[0][1], train_y)
    expected = "random_forest" if winning_index == 4 else "mean_baseline"
    assert table.attrs["selected_model"] == expected
    pd.testing.assert_frame_equal(calls[0] if calls else baseline_calls[0], test_x)
    assert table.drop(index=expected)[["test_mse", "test_r2"]].isna().all().all()


def test_final_date_boundary_cannot_split_same_day() -> None:
    features, target = pd.DataFrame({"size": range(12)}), pd.Series(range(12))
    dates = pd.Series(pd.date_range("2020-01-01", periods=4).repeat(3))
    with pytest.raises(ValueError, match="whole dates"):
        compare_regression_models(
            features, target, test_size=0.3, shuffle=False, dates=dates
        )
