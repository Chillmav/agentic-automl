"""CSV ingestion and explicit binary target validation."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    target: pd.Series
    fingerprint: str
    negative_class: str


def load_dataset(path: Path, target_column: str, positive_class: str) -> Dataset:
    """Hash and parse the same bytes; preserve target labels as strings."""
    content = path.read_bytes()
    frame = pd.read_csv(BytesIO(content), dtype={target_column: str})
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is absent from the CSV.")
    labels = frame[target_column]
    if labels.isna().any():
        raise ValueError("Target labels must not contain missing values.")
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise ValueError("Exactly two target classes are required.")
    if positive_class not in classes:
        raise ValueError(f"Positive class {positive_class!r} is absent; choose from {classes}.")
    features = frame.drop(columns=target_column)
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if labels.value_counts().min() < 5:
        raise ValueError("Each target class needs at least five rows for holdout and 3-fold CV.")
    return Dataset(
        features=features,
        target=labels.eq(positive_class).astype(int),
        fingerprint=sha256(content).hexdigest(),
        negative_class=next(label for label in classes if label != positive_class),
    )
