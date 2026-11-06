"""Local experiments: edit DATASETS to choose files and target columns."""

from pathlib import Path
from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split

from agentic_automl.comparison import compare_models
from agentic_automl.data import load_classification_csv

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


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    for filename, target_column in DATASETS.items():
        if target_column is None:
            print(f"Skipping {filename}: no binary target configured.")
            continue
        features, target = load_classification_csv(data_dir / filename, target_column)
        features = features.drop(columns=["person_id", "PassengerId"], errors="ignore")
        features, target = limit_samples(features, target)
        categorical = ["Pclass"] if filename == "Titanic-Dataset.xls" else None
        print(f"\n{filename}: {len(features):,} rows (including test rows)", flush=True)
        pprint(compare_models(features, target, categorical_columns=categorical))


if __name__ == "__main__":
    main()
