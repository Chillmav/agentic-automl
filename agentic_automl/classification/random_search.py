import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from agentic_automl.classification.validation import cross_validation_scores
from agentic_automl.optimization.random_search import random_search

SEARCH_SPACES = {
    "logistic_regression": {
        "estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "estimator__class_weight": [None, "balanced"],
    },
    "knn": {
        "estimator__n_neighbors": [3, 5, 7, 11],
        "estimator__weights": ["uniform", "distance"],
        "estimator__p": [1, 2],
    },
    "decision_tree": {
        "estimator__max_depth": [None, 3, 5, 10],
        "estimator__min_samples_leaf": [1, 3, 5],
        "estimator__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "estimator__n_estimators": [50, 100],
        "estimator__max_depth": [None, 5, 10],
        "estimator__min_samples_leaf": [1, 3, 5],
    },
    "svm": {
        "estimator__C": [0.1, 1.0, 10.0, 100.0],
        "estimator__gamma": ["scale", 0.01, 0.1, 1.0],
    },
}


def random_search_classifier(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    positive_class: str | int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    search_space: dict[str, list],
    n_trials: int = 5,
    default_scores: dict[str, float] | None = None,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    """Search development folds; optionally reuse this pipeline's scores on the same folds."""
    if n_trials < 1:
        raise ValueError("Random search requires at least 1 trial.")
    search_space = search_space.copy()
    if "estimator__n_neighbors" in search_space:
        smallest_fold = min(len(training) for training, _ in folds)
        search_space["estimator__n_neighbors"] = [
            k for k in search_space["estimator__n_neighbors"] if k <= smallest_fold
        ]
        if not search_space["estimator__n_neighbors"]:
            raise ValueError("No neighbor count fits the CV training folds.")
    return random_search(
        pipeline,
        search_space,
        lambda model: cross_validation_scores(
            model, features, target, positive_class, folds
        ),
        selection_metric="cv_f1_mean",
        direction="maximize",
        n_trials=n_trials,
        default_scores=default_scores,
    )
