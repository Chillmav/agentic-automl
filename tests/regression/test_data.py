from pathlib import Path

import pytest

from agentic_automl.regression.data import load_regression_csv


def test_load_regression_csv_preserves_numeric_target_and_missing_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.csv"
    path.write_text("size,city,price\n1,A,10.5\n,B,22.3\n")
    features, target = load_regression_csv(path, "price")
    assert features.columns.tolist() == ["size", "city"]
    assert features["size"].isna().tolist() == [False, True]
    assert target.tolist() == [10.5, 22.3]


@pytest.mark.parametrize(
    "content",
    [
        "size\n1\n",
        "price\n10\n",
        "size,price\n",
        "size,price\n1,\n",
        "size,price\n1,text\n",
        "size,price\n1,inf\n",
        "size,price\n1,True\n",
    ],
)
def test_invalid_regression_csv_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_regression_csv(path, "price")
