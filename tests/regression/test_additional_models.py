import numpy as np
import pandas as pd
import pytest

from agentic_automl.regression.models.decision_tree import train_decision_tree
from agentic_automl.regression.models.gradient_boosting import train_gradient_boosting
from agentic_automl.regression.models.random_forest import train_random_forest
from agentic_automl.regression.models.ridge import train_ridge

TRAINERS = [
    train_ridge,
    train_decision_tree,
    train_random_forest,
    train_gradient_boosting,
]


@pytest.mark.parametrize("trainer", TRAINERS)
def test_training_reuses_preprocessing_and_is_reproducible(trainer) -> None:
    features = pd.DataFrame(
        {
            "size": np.arange(40, dtype=float),
            "code": [1, 2] * 20,
            "identifier": [f"id-{i}" for i in range(40)],
        }
    )
    features.loc[0, "size"] = np.nan
    target = pd.Series(np.arange(40) * 2 + 1)
    model = trainer(features, target, categorical_columns=["code"])
    examples = pd.DataFrame({"size": [np.nan, 5.0], "code": [3, 1]})
    predictions = model.predict(examples)
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert model.named_steps["preprocessor"].rejected_columns_ == {"identifier": 40}
    repeated = trainer(features, target, categorical_columns=["code"])
    np.testing.assert_allclose(predictions, repeated.predict(examples))


@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("labels", [[1, None, 3], [1, np.inf, 3], ["a", "b", "c"]])
def test_invalid_targets_are_rejected(trainer, labels: list) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        trainer(pd.DataFrame({"size": [1, 2, 3]}), pd.Series(labels))


@pytest.mark.parametrize("trainer", TRAINERS)
def test_misaligned_rows_are_rejected(trainer) -> None:
    with pytest.raises(ValueError, match="matching row indices"):
        trainer(
            pd.DataFrame({"size": [1, 2, 3]}), pd.Series([1, 2, 3], index=[1, 2, 3])
        )
