import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from agentic_automl.preprocessing import build_pipeline


def train_svm(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Fit an RBF-kernel SVM; probability calibration stays disabled to save compute."""
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")
    pipeline = build_pipeline(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        categorical_columns=categorical_columns,
    )
    pipeline.fit(features, target)
    return pipeline
