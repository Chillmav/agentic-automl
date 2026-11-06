import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import TabularPreprocessor


def build_pipeline(
    estimator: BaseEstimator, categorical_columns: list[str] | None = None
) -> Pipeline:
    """Attach shared preprocessing to an estimator; fitting happens separately."""
    return Pipeline(
        [
            (
                "preprocessor",
                TabularPreprocessor(categorical_columns=categorical_columns),
            ),
            ("estimator", estimator),
        ]
    )


def train_logistic_regression(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit on training rows; categorical_columns overrides numeric dtype detection."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")

    pipeline = build_pipeline(
        LogisticRegression(random_state=42, max_iter=1000),
        categorical_columns=categorical_columns,
    )
    pipeline.fit(features, target)
    return pipeline
