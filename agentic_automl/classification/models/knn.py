import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import build_pipeline


def build_knn(categorical_columns: list[str] | None = None) -> Pipeline:
    """Construct an unfitted 5-neighbor pipeline."""
    return build_pipeline(
        KNeighborsClassifier(n_neighbors=5, n_jobs=1), categorical_columns
    )


def train_knn(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit 5-neighbor classification with shared preprocessing on training rows only."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")
    if len(features) < 5:
        raise ValueError("KNN with 5 neighbors requires at least 5 training rows.")
    pipeline = build_knn(categorical_columns)
    pipeline.fit(features, target)
    return pipeline
