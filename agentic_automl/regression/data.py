from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def validate_regression_target(target: pd.Series) -> None:
    """Regression requires finite numeric labels; missing labels aren't imputed."""
    if (
        not is_numeric_dtype(target)
        or is_bool_dtype(target)
        or target.isna().any()
        or not np.isfinite(target.to_numpy(dtype=float)).all()
    ):
        raise ValueError(
            "Regression target must contain finite numbers with no missing values."
        )


def load_regression_csv(
    path: str | Path, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    features, target = data.drop(columns=target_column), data[target_column]
    if features.shape[1] == 0 or data.empty:
        raise ValueError("At least one feature column and one data row are required.")
    validate_regression_target(target)
    return features, target
