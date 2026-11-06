from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_classification_csv(
    path: str | Path, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Load binary classification data as (features, target), preserving string labels."""
    data = pd.read_csv(path, dtype={target_column: str})
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    target = data[target_column]
    features = data.drop(columns=target_column)
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if target.isna().any():
        raise ValueError("Target labels must not be missing.")
    if target.nunique() != 2:
        raise ValueError("Exactly two target classes are required.")
    return features, target


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train features, test features, train labels, test labels before preprocessing."""
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Target must contain two classes and no missing labels.")
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    if train_target.nunique() != 2 or test_target.nunique() != 2:
        raise ValueError(
            "Both splits need two classes; use more data or adjust test_size."
        )
    return train_features, test_features, train_target, test_target
