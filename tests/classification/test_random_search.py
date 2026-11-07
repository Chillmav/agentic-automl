from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from agentic_automl.classification.models.decision_tree import build_decision_tree
from agentic_automl.classification.models.knn import build_knn
from agentic_automl.classification.models.logistic_regression import (
    build_logistic_regression,
)
from agentic_automl.classification.models.random_forest import build_random_forest
from agentic_automl.classification.models.svm import build_svm
from agentic_automl.classification.random_search import (
    SEARCH_SPACES,
    random_search_classifier,
)


def test_search_is_reproducible_selects_cv_best_and_returns_unfitted_model() -> None:
    features, target = pd.DataFrame({"value": range(30)}), pd.Series(["no", "yes"] * 15)
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=42).split(features, target)
    )
    outputs = []
    for _ in range(2):
        with patch(
            "agentic_automl.classification.random_search.cross_validation_scores",
            side_effect=[{"cv_f1_mean": 0.4}, {"cv_f1_mean": 0.8}],
        ) as validate:
            model, scores, trials = random_search_classifier(
                build_random_forest(),
                features,
                target,
                "yes",
                folds,
                SEARCH_SPACES["random_forest"],
                n_trials=2,
            )
        assert validate.call_count == 2
        for call in validate.call_args_list:
            assert call.args[4] is folds
            with pytest.raises(NotFittedError):
                check_is_fitted(call.args[0])
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
        assert scores["cv_f1_mean"] == 0.8
        for name, value in trials.loc[1, "parameters"].items():
            assert model.get_params()[name] == value
        assert trials.loc[0, "parameters"] != trials.loc[1, "parameters"]
        outputs.append(trials)
    pd.testing.assert_frame_equal(*outputs)


@pytest.mark.parametrize("n_trials", [0, -1])
def test_invalid_trial_budget_is_rejected(n_trials: int) -> None:
    with pytest.raises(ValueError, match="at least 1 trial"):
        random_search_classifier(
            build_random_forest(),
            pd.DataFrame({"value": range(10)}),
            pd.Series([0, 1] * 5),
            1,
            [],
            SEARCH_SPACES["random_forest"],
            n_trials=n_trials,
        )


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        ("logistic_regression", build_logistic_regression),
        ("knn", build_knn),
        ("decision_tree", build_decision_tree),
        ("random_forest", build_random_forest),
        ("svm", build_svm),
    ],
)
def test_each_classifier_can_be_searched(name, builder) -> None:
    features, target = pd.DataFrame({"value": range(30)}), pd.Series([0, 1] * 15)
    folds = list(StratifiedKFold(3).split(features, target))
    original = builder()
    with patch(
        "agentic_automl.classification.random_search.cross_validation_scores",
        side_effect=[{"cv_f1_mean": 0.4}, {"cv_f1_mean": 0.8}],
    ):
        model, scores, trials = random_search_classifier(
            original, features, target, 1, folds, SEARCH_SPACES[name], n_trials=2
        )
    assert scores["cv_f1_mean"] == 0.8
    for key, value in trials.loc[1, "parameters"].items():
        assert model.get_params()[key] == value
    assert (
        original.named_steps["estimator"].get_params()
        == builder().named_steps["estimator"].get_params()
    )


def test_knn_filters_infeasible_neighbors_and_caps_trial_budget() -> None:
    features, target = pd.DataFrame({"value": range(9)}), pd.Series([0, 1, 0] * 3)
    folds = list(StratifiedKFold(3).split(features, target))
    with patch(
        "agentic_automl.classification.random_search.cross_validation_scores",
        return_value={"cv_f1_mean": 0.5},
    ) as validate:
        _, _, trials = random_search_classifier(
            build_knn(), features, target, 1, folds, SEARCH_SPACES["knn"], n_trials=99
        )
    assert (
        validate.call_count == len(trials) == 8
    )  # Two feasible k, two weights, two distances.
    assert all(
        parameters["estimator__n_neighbors"] <= 6 for parameters in trials["parameters"]
    )
    assert SEARCH_SPACES["knn"]["estimator__n_neighbors"] == [3, 5, 7, 11]


def test_sampled_default_parameters_reuse_existing_cv_scores() -> None:
    features, target = pd.DataFrame({"value": range(30)}), pd.Series([0, 1] * 15)
    folds = list(StratifiedKFold(3).split(features, target))
    with patch(
        "agentic_automl.classification.random_search.cross_validation_scores",
        return_value={"cv_f1_mean": 0.7},
    ) as validate:
        model, scores, trials = random_search_classifier(
            build_random_forest(),
            features,
            target,
            1,
            folds,
            {"estimator__n_estimators": [50, 100]},
            n_trials=2,
            default_scores={"cv_f1_mean": 0.8},
        )
    assert (
        validate.call_count == 1
    )  # Only the non-default 50-tree configuration is fitted.
    assert model.get_params()["estimator__n_estimators"] == 100
    assert scores["cv_f1_mean"] == 0.8
    assert (
        len(trials) == 2
    )  # The reused configuration remains visible in the trial history.
