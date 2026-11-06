"""Regression experiments: run this file to display tables or call main() to return them."""

from pathlib import Path

import pandas as pd

from agentic_automl.classification.data import count_duplicate_rows
from agentic_automl.regression.comparison import compare_regression_models
from agentic_automl.regression.data import load_regression_csv

MAX_SAMPLES = 10_000
DATASETS = {
    "CAR DETAILS FROM CAR DEKHO.xls": "selling_price",  # CSV text.
    "uber.csv": "fare_amount",
    "Walmart_Sales.csv": "Weekly_Sales",
}


def limit_samples(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    sampled = features.sample(n=min(len(features), MAX_SAMPLES), random_state=42)
    return sampled, target.loc[sampled.index]


def main(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = (
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent / "data" / "regression"
    )
    tables = {}
    for filename, target_column in DATASETS.items():
        features, target = load_regression_csv(data_dir / filename, target_column)
        features = features.drop(columns=["Unnamed: 0", "key"], errors="ignore")
        features, target = limit_samples(features, target)
        temporal = filename == "Walmart_Sales.csv"
        test_size = 0.2
        dates = None
        if temporal:
            dates = pd.to_datetime(features["Date"], format="%d-%m-%Y").sort_values()
            order = dates.index
            features, target = features.loc[order], target.loc[order]
            cutoff = dates.drop_duplicates().iloc[int(dates.nunique() * 0.8)]
            test_size = int((dates >= cutoff).sum())
        table = compare_regression_models(
            features,
            target,
            categorical_columns=["Store", "Holiday_Flag"] if temporal else None,
            test_size=test_size,
            shuffle=not temporal,
            dates=dates,
        )
        table.attrs.update(
            sample_rows=len(features),
            duplicate_rows=count_duplicate_rows(features, target),
            split_strategy="chronological" if temporal else "random",
        )
        tables[filename] = table
    return tables


def print_results(tables: dict[str, pd.DataFrame]) -> None:
    """Display CV averages and a separate winner-only final test table."""
    for dataset, table in tables.items():
        print(f"\n{dataset} ({table.attrs['sample_rows']:,} rows)")
        print("CV averages:")
        print(
            table[["cv_mse_mean", "cv_r2_mean"]]
            .rename(columns={"cv_mse_mean": "mse", "cv_r2_mean": "r2"})
            .to_string()
        )
        winner = table.attrs["selected_model"]
        print(f"\nWinner selected by lowest mean CV MSE: {winner}\nFinal test results:")
        print(
            table.loc[[winner], ["test_mse", "test_r2"]]
            .rename(columns={"test_mse": "mse", "test_r2": "r2"})
            .to_string()
        )
        print(f"Rejected columns: {table.attrs['rejected_columns']}")


if __name__ == "__main__":
    print_results(main())
