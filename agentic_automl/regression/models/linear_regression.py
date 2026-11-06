import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline
from agentic_automl.regression.data import validate_regression_target


def train_linear_regression(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit shared preprocessing and ordinary least squares on training rows only."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    pipeline = build_pipeline(
        LinearRegression(), categorical_columns=categorical_columns
    )
    pipeline.fit(features, target)
    return pipeline
