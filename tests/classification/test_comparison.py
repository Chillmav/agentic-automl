from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.classification.data import load_classification_csv, stratified_split
from agentic_automl.classification.models.decision_tree import train_decision_tree
from agentic_automl.classification.models.knn import train_knn
from agentic_automl.classification.models.logistic_regression import (
    train_logistic_regression,
)
from agentic_automl.classification.models.random_forest import train_random_forest
from agentic_automl.classification.models.svm import train_svm
from agentic_automl.classification.validation import cross_validation_scores


def test_csv_to_comparison_beats_guessing_on_clear_signal(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"signal": [0] * 80 + [1] * 20, "target": ["no"] * 80 + ["yes"] * 20}
    ).to_csv(path, index=False)
    features, target = load_classification_csv(path, "target")
    results = compare_classification_models(features, target, positive_class="yes")
    test_columns = ["test_acc", "test_recall", "test_precision", "test_f1"]
    assert (
        results.columns.tolist()
        == [
            f"cv_{metric}_{summary}"
            for metric in ["acc", "recall", "precision", "f1"]
            for summary in ["mean", "std"]
        ]
        + test_columns
    )
    assert results.loc["logistic_regression", test_columns].eq(1.0).all()
    assert results.loc["knn", test_columns].eq(1.0).all()
    assert (
        results.loc[["decision_tree", "random_forest", "svm"], test_columns]
        .eq(1.0)
        .all()
        .all()
    )
    assert results.loc["guessing_baseline", "test_acc"] < 1.0
    assert results.loc["majority_baseline", test_columns].to_dict() == {
        "test_acc": 0.8,
        "test_recall": 0.0,
        "test_precision": 0.0,
        "test_f1": 0.0,
    }
    pd.testing.assert_frame_equal(
        results, compare_classification_models(features, target, positive_class="yes")
    )
    assert ((results >= 0) & (results <= 1)).all().all()

    negative_as_positive = compare_classification_models(
        features, target, positive_class="no"
    )
    assert negative_as_positive.loc["majority_baseline", "test_recall"] == 1.0
    assert negative_as_positive.loc["majority_baseline", "test_precision"] == 0.8
    assert negative_as_positive.loc["majority_baseline", "test_f1"] == pytest.approx(
        8 / 9
    )


@pytest.mark.parametrize(
    "trainer",
    [
        train_logistic_regression,
        train_knn,
        train_decision_tree,
        train_random_forest,
        train_svm,
    ],
)
def test_comparison_passes_only_training_rows_and_category_overrides(trainer) -> None:
    features = pd.DataFrame({"code": [1, 2] * 50})
    target = pd.Series(["no", "yes"] * 50)
    expected_train_x, _, expected_train_y, _ = stratified_split(
        features, target, test_size=0.3
    )
    with patch(
        f"agentic_automl.classification.comparison.{trainer.__name__}",
        wraps=trainer,
    ) as train:
        results = compare_classification_models(
            features,
            target,
            categorical_columns=["code"],
            test_size=0.3,
            positive_class="yes",
        )
    train.assert_called_once()
    pd.testing.assert_frame_equal(train.call_args.args[0], expected_train_x)
    pd.testing.assert_series_equal(train.call_args.args[1], expected_train_y)
    assert train.call_args.kwargs["categorical_columns"] == ["code"]
    assert results.loc["logistic_regression", "test_acc"] == 1.0


def test_unknown_positive_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Positive class.*missing"):
        compare_classification_models(
            pd.DataFrame({"value": range(20)}),
            pd.Series([0, 1] * 10),
            positive_class="missing",
        )


def test_all_models_use_shared_cv_folds_excluding_final_test_rows() -> None:
    features, target = pd.DataFrame({"value": range(60)}), pd.Series(["no", "yes"] * 30)
    expected_x, _, expected_y, _ = stratified_split(features, target)
    with patch(
        "agentic_automl.classification.comparison.cross_validation_scores",
        wraps=cross_validation_scores,
    ) as validate:
        compare_classification_models(features, target, positive_class="yes")
    assert validate.call_count == 7
    folds = validate.call_args_list[0].args[4]
    for call in validate.call_args_list:
        pd.testing.assert_frame_equal(call.args[1], expected_x)
        pd.testing.assert_series_equal(call.args[2], expected_y)
        assert call.args[4] is folds


@pytest.mark.parametrize(
    ("labels", "message"),
    [([0] * 17 + [1] * 3, "Each training class"), ([0, 1] * 4, "for KNN")],
)
def test_insufficient_cv_data_has_clear_error(labels: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_classification_models(
            pd.DataFrame({"value": range(len(labels))}),
            pd.Series(labels),
            positive_class=1,
        )
