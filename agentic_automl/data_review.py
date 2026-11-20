"""Deterministic suggestions for a user to review before training."""

import pandas as pd
from pandas.api.types import is_numeric_dtype


def review_dataset(data: pd.DataFrame, target_column: str) -> dict:
    changes, warnings = [], []
    missing = int(data[target_column].isna().sum())
    if missing:
        changes.append(
            {
                "kind": "drop_missing_targets",
                "column": target_column,
                "description": f"Exclude {missing} rows with missing target values from this run.",
            }
        )
    for column in data.columns.drop(target_column):
        values = data[column].dropna()
        if is_numeric_dtype(values) or values.empty:
            continue
        iso = values.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?")
        if iso.all():
            dates = pd.to_datetime(values, format="ISO8601", errors="coerce", utc=True)
            outputs = [
                f"{column}__{part}" for part in ("year", "month", "day", "weekday")
            ]
            if dates.notna().all() and not any(
                name in data.columns for name in outputs
            ):
                changes.append(
                    {
                        "kind": "expand_date",
                        "column": column,
                        "output_columns": outputs,
                        "description": f"Replace {column!r} with numeric year, month, day and weekday columns (UTC; time of day discarded).",
                    }
                )
            else:
                warnings.append(
                    f"{column!r}: invalid ISO dates or generated column names already exist; left unchanged."
                )
        elif iso.any():
            warnings.append(
                f"{column!r}: mixed date formats; left unchanged for manual inspection."
            )
        if values.nunique() > 20:
            warnings.append(
                f"{column!r} has more than 20 distinct values: without conversion, categorical preprocessing may reject it."
            )
    duplicates = int(data.duplicated().sum())
    if duplicates:
        warnings.append(
            f"Found {duplicates} duplicate rows; they will not be removed automatically."
        )
    warnings.append(
        "Training uses random splits. Dates or repeated entities may require a different split strategy; date expansion does not fix this."
    )
    return {
        "rows": len(data),
        "columns": [
            {
                "name": name,
                "dtype": str(data[name].dtype),
                "missing": int(data[name].isna().sum()),
                "distinct": int(data[name].nunique()),
            }
            for name in data.columns
        ],
        "proposed_changes": changes,
        "warnings": warnings,
    }


def apply_changes(data: pd.DataFrame, changes: list[dict]) -> pd.DataFrame:
    """Apply server-proposed operations to a copy; never alter the uploaded source."""
    prepared = data.copy()
    for change in changes:
        column = change["column"]
        if change["kind"] == "drop_missing_targets":
            prepared = prepared.loc[prepared[column].notna()].copy()
        elif change["kind"] == "expand_date":
            dates = pd.to_datetime(prepared[column], format="ISO8601", utc=True)
            parts = (dates.dt.year, dates.dt.month, dates.dt.day, dates.dt.dayofweek)
            for name, values in zip(change["output_columns"], parts):
                prepared[name] = values
            prepared = prepared.drop(columns=column)
        else:
            raise ValueError(f"Unknown operation: {change['kind']}")
    return prepared
