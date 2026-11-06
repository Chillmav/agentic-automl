import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline
from agentic_automl.regression.data import validate_regression_target


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit a 100-tree random forest on training rows only."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    pipeline = build_pipeline(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        categorical_columns,
    )
    pipeline.fit(features, target)
    return pipeline
