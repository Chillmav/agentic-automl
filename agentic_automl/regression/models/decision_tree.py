import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from agentic_automl.preprocessing import build_pipeline
from agentic_automl.regression.data import validate_regression_target


def build_decision_tree(categorical_columns: list[str] | None = None) -> Pipeline:
    """Construct an unfitted decision-tree regression pipeline."""
    return build_pipeline(DecisionTreeRegressor(random_state=42), categorical_columns)


def train_decision_tree(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit a decision tree with shared preprocessing on training rows only."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    pipeline = build_decision_tree(categorical_columns)
    pipeline.fit(features, target)
    return pipeline
