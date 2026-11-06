import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline
from agentic_automl.regression.data import validate_regression_target


def train_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit Ridge with alpha=1 using shared preprocessing on training rows only."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    pipeline = build_pipeline(Ridge(alpha=1.0, solver="lsqr"), categorical_columns)
    pipeline.fit(features, target)
    return pipeline
