import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from agentic_automl.preprocessing import build_pipeline


def build_decision_tree(categorical_columns: list[str] | None = None) -> Pipeline:
    """Construct an unfitted decision-tree pipeline."""
    return build_pipeline(DecisionTreeClassifier(random_state=42), categorical_columns)


def train_decision_tree(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit a decision tree using shared preprocessing on training rows only."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")
    pipeline = build_decision_tree(categorical_columns)
    pipeline.fit(features, target)
    return pipeline
