import numpy as np
import pandas as pd
import pytest

from agentic_automl.classification.models.knn import train_knn


def test_knn_predicts_with_numeric_missing_and_categorical_features() -> None:
    features = pd.DataFrame(
        {
            "size": [0, 0, np.nan, 0, 0, 10, 10, 10, 10, 10],
            "code": [1] * 5 + [2] * 5,
        }
    )
    target = pd.Series(["no"] * 5 + ["yes"] * 5)
    model = train_knn(features, target, categorical_columns=["code"])
    examples = pd.DataFrame({"size": [0, 10], "code": [1, 2]})
    assert model.predict(examples).tolist() == ["no", "yes"]
    probabilities = model.predict_proba(examples)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    repeated = train_knn(features, target, categorical_columns=["code"])
    np.testing.assert_allclose(probabilities, repeated.predict_proba(examples))


def test_knn_requires_enough_training_rows() -> None:
    with pytest.raises(ValueError, match="at least 5 training rows"):
        train_knn(pd.DataFrame({"size": range(4)}), pd.Series([0, 1, 0, 1]))


@pytest.mark.parametrize("labels", [[0] * 6, [0, 1, 2] * 2, [0, 1, None] * 2])
def test_knn_rejects_invalid_binary_targets(labels: list) -> None:
    with pytest.raises(ValueError, match="two classes"):
        train_knn(pd.DataFrame({"size": range(6)}), pd.Series(labels))
