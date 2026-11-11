import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from agentic_automl.regression.validation import (
    cross_validation_scores,
    regression_cv_folds,
)


def test_random_folds_are_reproducible_and_cover_validation_rows_once() -> None:
    features = pd.DataFrame({"size": range(30)})
    folds = regression_cv_folds(features)
    repeated = regression_cv_folds(features)
    assert len(folds) == 3
    assert sorted(np.concatenate([valid for _, valid in folds])) == list(range(30))
    for (train, valid), (other_train, other_valid) in zip(folds, repeated, strict=True):
        assert set(train).isdisjoint(valid)
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(valid, other_valid)


def test_temporal_cv_keeps_whole_dates_and_trains_before_validation() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=8).repeat(3))
    features = pd.DataFrame({"size": range(len(dates))})
    folds = regression_cv_folds(features, shuffle=False, dates=dates)
    assert len(folds) == 3
    for train, valid in folds:
        assert dates.iloc[train].max() < dates.iloc[valid].min()
        valid_dates = dates.iloc[valid].unique()
        assert set(valid) == set(np.flatnonzero(dates.isin(valid_dates)))


def test_cv_mse_is_positive_and_averages_fold_errors() -> None:
    features, target = pd.DataFrame({"size": range(12)}), pd.Series(range(12))
    folds = [(np.arange(4), np.arange(4, 8)), (np.arange(8), np.arange(8, 12))]
    model = DummyRegressor().fit(features, target)
    scores = cross_validation_scores(model, features, target, folds)
    errors = [
        float(((target.iloc[valid] - target.iloc[train].mean()) ** 2).mean())
        for train, valid in folds
    ]
    assert scores["cv_mse_mean"] == pytest.approx(np.mean(errors))
    assert scores["cv_mse_std"] == pytest.approx(np.std(errors))
    assert model.constant_[0, 0] == target.mean()


def test_cv_validation_requires_two_rows_for_r2() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        regression_cv_folds(pd.DataFrame({"size": range(5)}))


def test_temporal_dates_must_be_sorted() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=8)[::-1])
    with pytest.raises(ValueError, match="sorted"):
        regression_cv_folds(
            pd.DataFrame({"size": range(8)}), shuffle=False, dates=dates
        )
