import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from agentic_automl.data import load_dataset
from agentic_automl.experiment import run_experiment
from agentic_automl.ml import build_pipeline


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    features, target = make_classification(
        n_samples=150, n_features=5, n_informative=4, n_redundant=0, random_state=42
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(5)])
    frame["category"] = np.where(frame["feature_0"] > 0, "high", "low")
    frame.loc[::7, "category"] = np.nan
    frame.loc[::11, "feature_1"] = np.nan
    frame["target"] = np.where(target, "yes", "no")
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("frame", "positive_class", "message"),
    [
        ({"x": [1, 2]}, "yes", "absent"),
        ({"x": [1, 2], "target": ["yes", None]}, "yes", "missing"),
        ({"x": [1, 2, 3], "target": ["a", "b", "c"]}, "a", "two target classes"),
        ({"x": range(10), "target": ["a", "b"] * 5}, "yes", "Positive class"),
        ({"target": ["a", "b"] * 5}, "a", "feature column"),
        ({"x": range(6), "target": ["a", "b"] * 3}, "a", "five rows"),
    ],
)
def test_invalid_datasets(tmp_path: Path, frame: dict, positive_class: str, message: str) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_dataset(path, "target", positive_class)


def test_target_labels_preserve_leading_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pd.DataFrame({"x": range(10), "target": ["01", "02"] * 5}).to_csv(path, index=False)
    dataset = load_dataset(path, "target", "02")
    assert dataset.negative_class == "01"
    assert dataset.target.tolist() == [0, 1] * 5


def test_preprocessing_uses_only_training_rows_and_accepts_unseen_categories() -> None:
    features = pd.DataFrame({"number": [1.0, np.nan, 3.0, 5.0], "category": ["a", "b"] * 2})
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, [0, 1, 0, 1])
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == 3.0
    predictions = pipeline.predict_proba(pd.DataFrame({"number": [10000.0], "category": ["new"]}))
    assert predictions.shape == (1, 2)
    assert numeric.named_steps["impute"].statistics_[0] == 3.0


@pytest.mark.parametrize(
    "features",
    [
        pd.DataFrame({"flag": [True, False, True, False]}),
        pd.DataFrame({"category": ["a", "b", "a", "b"]}),
        pd.DataFrame({"number": [1.0, 2.0, 3.0, 4.0], "empty": [np.nan] * 4}),
    ],
)
def test_single_feature_types_and_empty_columns(features: pd.DataFrame) -> None:
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, [0, 1, 0, 1])
    assert np.isfinite(pipeline.predict_proba(features)).all()


def test_complete_run_is_reproducible_and_artifact_can_predict(
    csv_path: Path, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_experiment(csv_path, "target", "yes", first_dir)
    second = run_experiment(csv_path, "target", "yes", second_dir)
    assert first["candidates"] == second["candidates"]
    assert first["test_metrics"] == second["test_metrics"]
    assert first["selected_model"] == second["selected_model"]
    assert first["llm_cost_usd"] == 0
    assert first["candidates"][0]["validation"]["roc_auc"]["mean"] == 0.5
    assert first["test_metrics"]["roc_auc"] > 0.7
    manifest = json.loads((first_dir / "manifest.json").read_text())
    assert manifest == json.loads((second_dir / "manifest.json").read_text())
    train_indices = set(manifest["train_indices"])
    test_indices = set(manifest["test_indices"])
    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == set(range(150))
    validation_rows = []
    for fold in manifest["cv_indices"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert set(fold["train"]) | set(fold["validation"]) == train_indices
        validation_rows.extend(fold["validation"])
    assert sorted(validation_rows) == sorted(train_indices)
    dataset = load_dataset(csv_path, "target", "yes")
    pipeline = joblib.load(first_dir / "pipeline.joblib")
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    expected_medians = (
        dataset.features.iloc[manifest["train_indices"]].select_dtypes("number").median()
    )
    np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, expected_medians)
    probabilities = pipeline.predict_proba(dataset.features.iloc[manifest["test_indices"]])
    assert probabilities.shape == (len(test_indices), 2)
    assert np.isfinite(probabilities).all()


def test_existing_results_are_not_overwritten(csv_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("existing results")
    with pytest.raises(FileExistsError):
        run_experiment(csv_path, "target", "yes", output)
    assert marker.read_text() == "existing results"


def test_cli(csv_path: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_automl",
            str(csv_path),
            "--target",
            "target",
            "--positive-class",
            "yes",
            "--output",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["selected_model"] in {"dummy", "logistic_regression"}


def test_cli_reports_invalid_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_automl",
            str(tmp_path / "absent.csv"),
            "--target",
            "target",
            "--positive-class",
            "yes",
            "--output",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Error:" in result.stderr
