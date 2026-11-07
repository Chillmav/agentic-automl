from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from agentic_automl.regression.models.decision_tree import build_decision_tree
from agentic_automl.regression.models.gradient_boosting import build_gradient_boosting
from agentic_automl.regression.models.linear_regression import build_linear_regression
from agentic_automl.regression.models.random_forest import build_random_forest
from agentic_automl.regression.models.ridge import build_ridge
from agentic_automl.regression.random_search import (
    SEARCH_SPACES,
    random_search_regressor,
)


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        ("linear_regression", build_linear_regression),
        ("ridge", build_ridge),
        ("decision_tree", build_decision_tree),
        ("random_forest", build_random_forest),
        ("gradient_boosting", build_gradient_boosting),
    ],
)
def test_each_regressor_selects_lowest_cv_mse(name, builder) -> None:
    features, target = pd.DataFrame({"value": range(30)}), pd.Series(range(30))
    folds = list(KFold(3).split(features))
    with patch(
        "agentic_automl.regression.random_search.cross_validation_scores",
        side_effect=[
            {"cv_mse_mean": 2.0, "cv_r2_mean": 0.9},
            {"cv_mse_mean": 1.0, "cv_r2_mean": 0.2},
        ],
    ) as validate:
        model, scores, trials = random_search_regressor(
            builder(), features, target, folds, SEARCH_SPACES[name], n_trials=2
        )
    assert scores["cv_mse_mean"] == 1.0
    for key, value in trials.loc[1, "parameters"].items():
        assert model.get_params()[key] == value
    for call in validate.call_args_list:
        assert call.args[3] is folds
        pd.testing.assert_frame_equal(call.args[1], features)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_regression_search_reuses_default_scores() -> None:
    features, target = pd.DataFrame({"value": range(30)}), pd.Series(range(30))
    folds = list(KFold(3).split(features))
    with patch(
        "agentic_automl.regression.random_search.cross_validation_scores",
        return_value={"cv_mse_mean": 2.0},
    ) as validate:
        model, scores, table = random_search_regressor(
            build_ridge(),
            features,
            target,
            folds,
            {"estimator__alpha": [1.0, 10.0]},
            n_trials=2,
            default_scores={"cv_mse_mean": 1.0},
        )
    assert validate.call_count == 1
    assert len(table) == 2
    assert model.get_params()["estimator__alpha"] == 1.0
    assert scores["cv_mse_mean"] == 1.0
