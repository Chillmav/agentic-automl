import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from agentic_automl.preprocessing import TabularPreprocessor


def test_only_categorical_columns_above_20_values_are_rejected() -> None:
    features = pd.DataFrame(
        {
            "number": range(21),
            "keep": [str(i) for i in range(20)] + [None],
            "reject": [str(i) for i in range(21)],
        }
    )
    preprocessor = TabularPreprocessor().fit(features)
    assert preprocessor.rejected_columns_ == {"reject": 21}
    assert preprocessor.transform(features).shape == (21, 21)
    # Selection is fixed at fit time; rejected columns aren't required at prediction.
    new_data = features.drop(columns="reject").copy()
    new_data["keep"] = [f"unseen-{i}" for i in range(21)]
    assert preprocessor.transform(new_data).shape == (21, 21)
    assert preprocessor.rejected_columns_ == {"reject": 21}
    assert not hasattr(clone(preprocessor), "rejected_columns_")


@pytest.mark.parametrize("dtype", ["object", "string", "category", "bool"])
def test_categorical_dtypes_are_supported(dtype: str) -> None:
    features = pd.DataFrame({"category": pd.Series([True, False, True], dtype=dtype)})
    encoded = TabularPreprocessor().fit_transform(features)
    assert encoded.shape == (3, 2)
    assert np.isfinite(encoded).all()


def test_no_usable_columns_has_clear_error() -> None:
    features = pd.DataFrame({"identifier": [str(i) for i in range(21)]})
    with pytest.raises(ValueError, match="identifier.*21"):
        TabularPreprocessor().fit(features)


@pytest.mark.parametrize("distinct_values", [20, 21])
def test_numeric_category_override_uses_category_limit(distinct_values: int) -> None:
    features = pd.DataFrame(
        {"code": range(distinct_values), "size": range(distinct_values)}
    )
    preprocessor = TabularPreprocessor(categorical_columns=["code"]).fit(features)
    expected_rejections = {"code": 21} if distinct_values == 21 else {}
    assert preprocessor.rejected_columns_ == expected_rejections
    assert preprocessor.transform(features).shape[1] == (
        1 if distinct_values == 21 else 21
    )
    assert clone(preprocessor).categorical_columns == ["code"]


def test_unknown_categorical_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Categorical columns.*typo"):
        TabularPreprocessor(categorical_columns=["typo"]).fit(
            pd.DataFrame({"size": [1, 2]})
        )
