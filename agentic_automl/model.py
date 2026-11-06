import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agentic_automl.preprocessing import TabularPreprocessor


def train_logistic_regression(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit on training rows; categorical_columns overrides numeric dtype detection."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                TabularPreprocessor(categorical_columns=categorical_columns),
            ),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    pipeline.fit(features, target)
    return pipeline
