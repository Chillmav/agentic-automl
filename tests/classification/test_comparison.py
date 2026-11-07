from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from agentic_automl.classification.comparison import compare_classification_models
from agentic_automl.classification.data import load_classification_csv, stratified_split
from agentic_automl.classification.models.decision_tree import build_decision_tree
from agentic_automl.classification.models.knn import build_knn
from agentic_automl.classification.models.logistic_regression import (
    build_logistic_regression,
)
from agentic_automl.classification.models.random_forest import build_random_forest
from agentic_automl.classification.models.svm import build_svm
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
    assert results.attrs["selected_model"] == "logistic_regression"
    assert results.drop(index="logistic_regression")[test_columns].isna().all().all()
    assert results.loc["majority_baseline", "cv_f1_mean"] == 0.0
    pd.testing.assert_frame_equal(
        results, compare_classification_models(features, target, positive_class="yes")
    )
    assert (((results >= 0) & (results <= 1)) | results.isna()).all().all()

    negative_as_positive = compare_classification_models(
        features, target, positive_class="no"
    )
    assert negative_as_positive.loc["majority_baseline", "cv_recall_mean"] == 1.0
    assert negative_as_positive.loc[
        "majority_baseline", "cv_precision_mean"
    ] == pytest.approx(0.8, abs=0.01)
    assert negative_as_positive.loc["logistic_regression", test_columns].eq(1.0).all()


@pytest.mark.parametrize(
    "trainer",
    [
        build_logistic_regression,
        build_knn,
        build_decision_tree,
        build_random_forest,
        build_svm,
    ],
)
def test_comparison_passes_category_overrides_to_builders(trainer) -> None:
    features = pd.DataFrame({"code": [1, 2] * 50})
    target = pd.Series(["no", "yes"] * 50)
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
    train.assert_called_once_with(categorical_columns=["code"])
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


@pytest.mark.parametrize("winning_index", [0, 5])
def test_only_cv_winner_predicts_test_rows_with_deterministic_ties(
    winning_index: int,
) -> None:
    features, target = pd.DataFrame({"value": range(60)}), pd.Series(["no", "yes"] * 30)
    train_x, test_x, train_y, test_y = stratified_split(features, target)
    cv_scores = [
        {"cv_f1_mean": 1.0 if index == winning_index == 5 else 0.5}
        for index in range(7)
    ]
    pipeline_predictions = []
    fits = []
    original_pipeline_fit = Pipeline.fit
    original_baseline_fit = DummyClassifier.fit

    def fit_pipeline(model, fit_features, fit_target):
        fits.append((fit_features, fit_target))
        return original_pipeline_fit(model, fit_features, fit_target)

    def fit_baseline(model, fit_features, fit_target):
        fits.append((fit_features, fit_target))
        return original_baseline_fit(model, fit_features, fit_target)

    def predict_pipeline(model, test_features):
        pipeline_predictions.append((model, test_features))
        return test_y.to_numpy()

    with (
        patch(
            "agentic_automl.classification.comparison.cross_validation_scores",
            side_effect=cv_scores,
        ),
        patch.object(Pipeline, "predict", new=predict_pipeline),
        patch.object(Pipeline, "fit", new=fit_pipeline),
        patch.object(DummyClassifier, "fit", new=fit_baseline),
        patch.object(
            DummyClassifier, "predict", autospec=True, return_value=test_y.to_numpy()
        ) as baseline_predict,
    ):
        table = compare_classification_models(features, target, positive_class="yes")
    assert len(pipeline_predictions) + baseline_predict.call_count == 1
    assert len(fits) == 1
    pd.testing.assert_frame_equal(fits[0][0], train_x)
    pd.testing.assert_series_equal(fits[0][1], train_y)
    if winning_index == 5:
        assert table.attrs["selected_model"] == "random_forest"
        assert isinstance(
            pipeline_predictions[0][0].named_steps["estimator"],
            RandomForestClassifier,
        )
        pd.testing.assert_frame_equal(pipeline_predictions[0][1], test_x)
    else:
        assert (
            table.attrs["selected_model"] == "guessing_baseline"
        )  # First model wins a CV tie.
        pd.testing.assert_frame_equal(baseline_predict.call_args.args[1], test_x)
    test_scores = table.filter(like="test_")
    assert test_scores.notna().all(axis=1).sum() == 1
    assert test_scores.drop(index=table.attrs["selected_model"]).isna().all().all()
