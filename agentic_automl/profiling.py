"""Small descriptive profiles computed only from the outer training split."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def training_profile(features: pd.DataFrame, target: pd.Series, task: str) -> dict:
    columns = []
    for name in features.columns:
        values = features[name]
        column = {
            "name": name,
            "dtype": str(values.dtype),
            "missing_percent": float(values.isna().mean() * 100),
            "distinct": int(values.nunique()),
        }
        if is_numeric_dtype(values):
            numeric = values.replace([np.inf, -np.inf], np.nan).dropna()
            column["numeric"] = {
                key: float(value) if pd.notna(value) else None
                for key, value in {
                    "min": numeric.min(),
                    "max": numeric.max(),
                    "mean": numeric.mean(),
                    "median": numeric.median(),
                }.items()
            }
        columns.append(column)
    if task == "classification":
        target_summary = {
            "class_counts": {
                str(label): int(count) for label, count in target.value_counts().items()
            }
        }
    else:
        target_summary = {
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "median": float(target.median()),
        }
    return {
        "scope": "outer_training_split",
        "rows": len(features),
        "feature_count": len(columns),
        "columns": columns,
        "target": target_summary,
    }
