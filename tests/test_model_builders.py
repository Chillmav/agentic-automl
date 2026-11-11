import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from agentic_automl.classification.models.decision_tree import (
    build_decision_tree as classifier_tree,
)
from agentic_automl.classification.models.knn import build_knn
from agentic_automl.classification.models.logistic_regression import (
    build_logistic_regression,
)
from agentic_automl.classification.models.random_forest import (
    build_random_forest as classifier_forest,
)
from agentic_automl.classification.models.svm import build_svm
from agentic_automl.regression.models.decision_tree import (
    build_decision_tree as regressor_tree,
)
from agentic_automl.regression.models.gradient_boosting import build_gradient_boosting
from agentic_automl.regression.models.linear_regression import build_linear_regression
from agentic_automl.regression.models.random_forest import (
    build_random_forest as regressor_forest,
)
from agentic_automl.regression.models.ridge import build_ridge


@pytest.mark.parametrize(
    "builder",
    [
        build_logistic_regression,
        build_knn,
        classifier_tree,
        classifier_forest,
        build_svm,
        build_linear_regression,
        build_ridge,
        regressor_tree,
        regressor_forest,
        build_gradient_boosting,
    ],
)
def test_builder_is_unfitted_and_cloneable(builder) -> None:
    model = builder(categorical_columns=["code"])
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert clone(model).named_steps["preprocessor"].categorical_columns == ["code"]


def test_tuning_parameters_changes_only_the_new_pipeline() -> None:
    model = classifier_forest().set_params(
        estimator__max_depth=3, estimator__n_estimators=10
    )
    assert model.named_steps["estimator"].max_depth == 3
    assert model.named_steps["estimator"].n_estimators == 10
    assert classifier_forest().named_steps["estimator"].n_estimators == 100
