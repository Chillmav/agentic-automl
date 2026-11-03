"""A deterministic binary classification experiment and its local artifacts."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from agentic_automl.data import load_dataset
from agentic_automl.ml import build_pipeline, evaluate

METRICS = ("roc_auc", "average_precision", "balanced_accuracy", "f1")


def run_experiment(
    csv_path: Path,
    target_column: str,
    positive_class: str,
    output_dir: Path,
    random_state: int = 42,
) -> dict:
    """Select by development CV ROC-AUC, then evaluate the winner once on test_data."""
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new path.")
    dataset = load_dataset(csv_path, target_column, positive_class)
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset.target)),
        test_size=0.2,
        stratify=dataset.target,
        random_state=random_state,
    )
    x_train = dataset.features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    x_test = dataset.features.iloc[test_indices]
    y_test = dataset.target.iloc[test_indices]
    if y_train.value_counts().min() < 3 or y_test.nunique() != 2:
        raise ValueError("Insufficient class support for stratified test_data and 3-fold CV.")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    folds = list(cv.split(x_train, y_train))
    candidates = []
    for name in ("dummy", "logistic_regression"):
        scores = cross_validate(
            build_pipeline(name, random_state),
            x_train,
            y_train,
            cv=folds,
            scoring=list(METRICS),
            n_jobs=1,
            error_score="raise",
        )
        candidates.append(
            {
                "model": name,
                "validation": {
                    metric: {
                        "mean": float(np.mean(scores[f"test_{metric}"])),
                        "std": float(np.std(scores[f"test_{metric}"])),
                        "fold_scores": scores[f"test_{metric}"].tolist(),
                    }
                    for metric in METRICS
                },
            }
        )
    winner = max(candidates, key=lambda item: item["validation"]["roc_auc"]["mean"])
    pipeline = build_pipeline(winner["model"], random_state)
    pipeline.fit(x_train, y_train)
    test_metrics = evaluate(pipeline, x_test, y_test)
    report = {
        "selected_model": winner["model"],
        "selection_metric": "roc_auc",
        "candidates": candidates,
        "test_metrics": test_metrics,
        "elapsed_seconds": perf_counter() - started,
        "llm_cost_usd": 0.0,
    }
    manifest = {
        "dataset_sha256": dataset.fingerprint,
        "dataset_path": str(csv_path.resolve()),
        "target_column": target_column,
        "class_mapping": {"0": dataset.negative_class, "1": positive_class},
        "feature_columns": dataset.features.columns.tolist(),
        "feature_dtypes": {name: str(dtype) for name, dtype in dataset.features.dtypes.items()},
        "rows": len(dataset.target),
        "random_state": random_state,
        "test_fraction": 0.2,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "cv_indices": [
            {
                "train": train_indices[train].tolist(),
                "validation": train_indices[validation].tolist(),
            }
            for train, validation in folds
        ],
        "python_version": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "joblib",
                "scipy",
                "threadpoolctl",
                "python-dateutil",
                "pytz",
                "tzdata",
            )
        },
        "selected_model": winner["model"],
        "pipeline_parameters": {
            name: value
            for name, value in pipeline.get_params(deep=True).items()
            if isinstance(value, str | int | float | bool | type(None))
        },
    }
    # Reserve a fresh directory to avoid overwriting existing experiment artifacts.
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report
