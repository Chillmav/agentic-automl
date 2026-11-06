from pathlib import Path

import pandas as pd
import pytest

from agentic_automl.classification.data import load_classification_csv, stratified_split
from agentic_automl.classification.models.logistic_regression import (
    train_logistic_regression,
)


@pytest.mark.parametrize("test_size", [0.2, 0.3])
def test_loaded_data_splits_preserving_class_proportions(
    tmp_path: Path, test_size: float
) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"value": range(100), "target": ["no"] * 80 + ["yes"] * 20}).to_csv(
        path, index=False
    )
    features, target = load_classification_csv(path, "target")
    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=test_size
    )
    assert len(test_x) == int(100 * test_size)
    assert len(train_x) + len(test_x) == 100
    assert set(train_x.index).isdisjoint(test_x.index)
    assert set(train_x.index) | set(test_x.index) == set(features.index)
    for split_x, split_y in [(train_x, train_y), (test_x, test_y)]:
        pd.testing.assert_series_equal(split_y, target.loc[split_x.index])
        assert split_y.eq("yes").mean() == pytest.approx(0.2)
    repeated = stratified_split(features, target, test_size=test_size)
    pd.testing.assert_frame_equal(train_x, repeated[0])
    pd.testing.assert_frame_equal(test_x, repeated[1])
    model = train_logistic_regression(train_x, train_y)
    assert len(model.predict(test_x)) == len(test_y)


def test_misaligned_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="matching row indices"):
        stratified_split(
            pd.DataFrame({"value": range(10)}),
            pd.Series([0, 1] * 5, index=range(1, 11)),
        )


def test_insufficient_class_support_is_rejected() -> None:
    with pytest.raises(ValueError):
        stratified_split(pd.DataFrame({"value": range(10)}), pd.Series([0] * 9 + [1]))


def test_split_without_both_classes_is_rejected() -> None:
    with pytest.raises(ValueError, match="Both splits need two classes"):
        stratified_split(
            pd.DataFrame({"value": range(100)}), pd.Series([0] * 98 + [1] * 2)
        )
