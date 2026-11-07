import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline


def build_random_forest(categorical_columns: list[str] | None = None) -> Pipeline:
    """Construct an unfitted 100-tree random-forest pipeline."""
    return build_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        categorical_columns,
    )


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit a 100-tree random forest on training rows only."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")
    pipeline = build_random_forest(categorical_columns)
    pipeline.fit(features, target)
    return pipeline
