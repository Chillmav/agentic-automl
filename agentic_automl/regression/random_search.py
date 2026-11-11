import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from agentic_automl.optimization.random_search import random_search
from agentic_automl.regression.validation import cross_validation_scores

SEARCH_SPACES = {
    "linear_regression": {"estimator__fit_intercept": [True, False]},
    "ridge": {
        "estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "estimator__fit_intercept": [True, False],
    },
    "decision_tree": {
        "estimator__max_depth": [None, 3, 5, 10],
        "estimator__min_samples_leaf": [1, 3, 5],
        "estimator__max_features": [None, "sqrt"],
    },
    "random_forest": {
        "estimator__n_estimators": [50, 100],
        "estimator__max_depth": [None, 5, 10],
        "estimator__min_samples_leaf": [1, 3, 5],
    },
    "gradient_boosting": {
        "estimator__n_estimators": [50, 100],
        "estimator__learning_rate": [0.03, 0.1, 0.2],
        "estimator__max_depth": [2, 3, 5],
        "estimator__min_samples_leaf": [1, 3],
    },
}


def random_search_regressor(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    search_space: dict[str, list],
    n_trials: int = 5,
    default_scores: dict[str, float] | None = None,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    """Search only development folds, including unchanged chronological folds."""
    return random_search(
        pipeline,
        search_space,
        lambda model: cross_validation_scores(model, features, target, folds),
        selection_metric="cv_mse_mean",
        direction="minimize",
        n_trials=n_trials,
        default_scores=default_scores,
    )
