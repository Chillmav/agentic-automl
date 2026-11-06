import numpy as np
import pandas as pd

from agentic_automl.regression.models.linear_regression import train_linear_regression


def test_linear_regression_learns_line() -> None:
    model = train_linear_regression(
        pd.DataFrame({"size": [0, 1, 2, 3]}), pd.Series([1, 3, 5, 7])
    )
    np.testing.assert_allclose(model.predict(pd.DataFrame({"size": [4, 5]})), [9, 11])


def test_regression_reuses_categorical_overrides_and_imputation() -> None:
    features = pd.DataFrame(
        {"code": [1, 2, 1, 2, 1, 2], "size": [1, 2, np.nan, 4, 5, 6]}
    )
    model = train_linear_regression(features, pd.Series([1, 4, 3, 8, 5, 12]), ["code"])
    preprocessor = model.named_steps["preprocessor"]
    assert preprocessor.transform(features).shape == (6, 3)
    assert np.isfinite(
        model.predict(pd.DataFrame({"code": [3], "size": [np.nan]}))
    ).all()
