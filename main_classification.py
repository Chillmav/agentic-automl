"""Local experiments: edit DATASETS to choose files and target columns."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.classification.data import (
    count_duplicate_rows,
    load_classification_csv,
)

MAX_SAMPLES = 10_000
DATASETS = {
    "diabetes_prediction_dataset.csv": "diabetes",
    "student_dropout_risk_2026.csv": "dropout",
    "Titanic-Dataset.xls": "Survived",  # CSV text despite its extension.
    "airlines.csv": None,  # No binary target available.
}


def limit_samples(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    if len(features) <= MAX_SAMPLES:
        return features, target
    sampled_x, _, sampled_y, _ = train_test_split(
        features, target, train_size=MAX_SAMPLES, stratify=target, random_state=42
    )
    return sampled_x, sampled_y


def main(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Return one score table per dataset; sample and duplicate counts live in table.attrs."""
    data_dir = (
        data_dir
        if data_dir is not None
        else Path(__file__).resolve().parent / "data" / "binary_classification"
    )
    tables = {}
    for filename, target_column in DATASETS.items():
        if target_column is None:
            continue
        features, target = load_classification_csv(data_dir / filename, target_column)
        features = features.drop(columns=["person_id", "PassengerId"], errors="ignore")
        features, target = limit_samples(features, target)
        categorical = ["Pclass"] if filename == "Titanic-Dataset.xls" else None
        table = compare_classification_models(
            features, target, categorical_columns=categorical, positive_class="1"
        )
        table.attrs.update(
            sample_rows=len(features),
            duplicate_rows=count_duplicate_rows(features, target),
        )
        tables[filename] = table
    return tables


def print_results(tables: dict[str, pd.DataFrame]) -> None:
    """Display CV means for all models, followed by the winner's test scores."""
    for dataset, table in tables.items():
        print(f"\n{dataset}\nCV averages:")
        averages = table.filter(regex=r"^cv_.*_mean$").rename(
            columns=lambda name: name.removeprefix("cv_").removesuffix("_mean")
        )
        print(averages.to_string())
        winner = table.attrs["selected_model"]
        print(f"\nWinner selected by mean CV F1: {winner}\nFinal test results:")
        test_scores = (
            table.filter(regex=r"^test_")
            .loc[[winner]]
            .rename(columns=lambda name: name.removeprefix("test_"))
        )
        print(test_scores.to_string())


if __name__ == "__main__":
    print_results(main())
