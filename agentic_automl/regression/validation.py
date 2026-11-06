import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate


def regression_cv_folds(
    features: pd.DataFrame, shuffle: bool = True, dates: pd.Series | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shared 3-fold splits, keeping whole dates together for temporal data."""
    if dates is not None:
        if shuffle or not dates.index.equals(features.index):
            raise ValueError(
                "Dates require ordered rows, matching indices, and shuffle=False."
            )
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise ValueError(
                "Dates must be non-missing and sorted from earliest to latest."
            )
        unique_dates = dates.unique()
        folds = [
            (
                np.flatnonzero(dates.isin(unique_dates[train])),
                np.flatnonzero(dates.isin(unique_dates[valid])),
            )
            for train, valid in TimeSeriesSplit(n_splits=3).split(unique_dates)
        ]
    else:
        splitter = (
            KFold(3, shuffle=True, random_state=42) if shuffle else TimeSeriesSplit(3)
        )
        folds = list(splitter.split(features))
    if any(len(valid) < 2 for _, valid in folds):
        raise ValueError("Each CV validation fold needs at least two rows for R2.")
    return folds


def cross_validation_scores(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Refit each pipeline per fold; report positive MSE and R2 mean/std."""
    scores = cross_validate(
        model,
        features,
        target,
        cv=folds,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=1,
        error_score="raise",
    )
    return {
        f"cv_{metric}_{summary}": float(aggregate(sign * scores[f"test_{metric}"]))
        for metric, sign in [("mse", -1), ("r2", 1)]
        for summary, aggregate in [("mean", np.mean), ("std", np.std)]
    }
