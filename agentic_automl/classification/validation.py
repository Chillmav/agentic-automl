import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate


def cross_validation_scores(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    positive_class: str | int,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Clone and refit the entire model pipeline inside each development fold."""
    scoring = {"acc": "accuracy"}
    for name, metric in [
        ("recall", recall_score),
        ("precision", precision_score),
        ("f1", f1_score),
    ]:
        scoring[name] = make_scorer(metric, pos_label=positive_class, zero_division=0)
    scores = cross_validate(
        model,
        features,
        target,
        cv=folds,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return {
        f"cv_{metric}_{summary}": float(aggregate(scores[f"test_{metric}"]))
        for metric in scoring
        for summary, aggregate in [("mean", np.mean), ("std", np.std)]
    }
