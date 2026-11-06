from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_automl.data import load_classification_csv
from agentic_automl.model import train_logistic_regression


def test_loaded_csv_can_train_and_predict(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("size,target\n0,no\n1,no\n2,no\n8,yes\n9,yes\n10,yes\n")
    features, target = load_classification_csv(path, "target")
    model = train_logistic_regression(features, target)

    examples = pd.DataFrame({"size": [-2, 12]})
    assert model.predict(examples).tolist() == ["no", "yes"]
    probabilities = model.predict_proba(examples)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    repeated = train_logistic_regression(features, target)
    np.testing.assert_allclose(probabilities, repeated.predict_proba(examples))


def test_missing_features_are_imputed_from_training_data() -> None:
    features = pd.DataFrame({"size": [0, 1, np.nan, 8, 9, 10]})
    target = pd.Series(["no"] * 3 + ["yes"] * 3)
    model = train_logistic_regression(features, target)

    numeric = model.named_steps["preprocessor"].transformer_.named_transformers_[
        "numeric"
    ]
    assert numeric.named_steps["imputer"].statistics_[0] == 8
    assert np.isfinite(model.predict_proba(features)).all()


def test_categorical_features_can_train_and_predict() -> None:
    model = train_logistic_regression(
        pd.DataFrame({"city": ["A", "B", None, "A", "B", "A"]}),
        pd.Series([0, 1, 0, 0, 1, 0]),
    )
    assert model.predict(pd.DataFrame({"city": ["A", "B"]})).tolist() == [0, 1]


def test_training_passes_numeric_category_override_to_preprocessor() -> None:
    features = pd.DataFrame({"code": [1, 2, 3, 1, 2, 3]})
    model = train_logistic_regression(
        features, pd.Series([0, 1, 0, 0, 1, 0]), categorical_columns=["code"]
    )
    preprocessor = model.named_steps["preprocessor"]
    assert preprocessor.transform(features).shape == (6, 3)
    assert model.predict(pd.DataFrame({"code": [1, 2]})).tolist() == [0, 1]


@pytest.mark.parametrize("labels", [[0, 0, 0], [0, 1, 2], [0, 1, None]])
def test_invalid_target_is_rejected(labels: list) -> None:
    with pytest.raises(ValueError, match="two classes"):
        train_logistic_regression(pd.DataFrame({"size": [1, 2, 3]}), pd.Series(labels))
