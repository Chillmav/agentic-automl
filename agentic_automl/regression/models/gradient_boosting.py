import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline
from agentic_automl.regression.data import validate_regression_target


def build_gradient_boosting(categorical_columns: list[str] | None = None) -> Pipeline:
    """Construct an unfitted 100-stage gradient-boosting regression pipeline."""
    return build_pipeline(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        categorical_columns,
    )


def train_gradient_boosting(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit 100 boosting stages with shared preprocessing on training rows only."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    pipeline = build_gradient_boosting(categorical_columns)
    pipeline.fit(features, target)
    return pipeline
