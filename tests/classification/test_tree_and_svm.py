import numpy as np
import pandas as pd
import pytest

from agentic_automl.classification.models.decision_tree import train_decision_tree
from agentic_automl.classification.models.random_forest import train_random_forest
from agentic_automl.classification.models.svm import train_svm

TRAINERS = [train_decision_tree, train_random_forest, train_svm]


@pytest.mark.parametrize("trainer", TRAINERS)
def test_prediction_preprocessing_and_reproducibility(trainer) -> None:
    features = pd.DataFrame(
        {
            "size": [0.0] * 20 + [10.0] * 20,
            "code": [1] * 20 + [2] * 20,
            "identifier": [f"id-{i}" for i in range(40)],
        }
    )
    features.loc[0, "size"] = np.nan
    target = pd.Series(["no"] * 20 + ["yes"] * 20)
    model = trainer(features, target, categorical_columns=["code"])
    examples = pd.DataFrame({"size": [0.0, 10.0], "code": [1, 2]})
    assert model.predict(examples).tolist() == ["no", "yes"]
    assert model.named_steps["preprocessor"].rejected_columns_ == {"identifier": 40}
    unseen = pd.DataFrame({"size": [np.nan], "code": [3]})
    assert model.predict(unseen)[0] in ["no", "yes"]
    repeated = trainer(features, target, categorical_columns=["code"])
    np.testing.assert_array_equal(model.predict(features), repeated.predict(features))


@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("labels", [[0] * 6, [0, 1, 2] * 2, [0, 1, None] * 2])
def test_invalid_binary_targets_are_rejected(trainer, labels: list) -> None:
    with pytest.raises(ValueError, match="two classes"):
        trainer(pd.DataFrame({"size": range(6)}), pd.Series(labels))
