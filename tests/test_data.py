from pathlib import Path

import pytest

from agentic_automl.data import load_classification_csv


def test_load_csv_separates_features_and_preserves_labels(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("age,city,target\n20,A,01\n,B,02\n", encoding="utf-8")

    features, target = load_classification_csv(path, "target")

    assert features.columns.tolist() == ["age", "city"]
    assert features.shape == (2, 2)
    assert features["age"].isna().tolist() == [False, True]
    assert features["city"].tolist() == ["A", "B"]
    assert target.tolist() == ["01", "02"]
    assert target.name == "target"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("age\n20\n", "was not found"),
        ("target\nyes\nno\n", "feature column"),
        ("age,target\n20,yes\n30,\n", "must not be missing"),
        ("age,target\n20,yes\n30,yes\n", "two target classes"),
        ("age,target\n20,a\n30,b\n40,c\n", "two target classes"),
        ("age,target\n", "two target classes"),
    ],
)
def test_invalid_data_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_classification_csv(path, "target")


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classification_csv(tmp_path / "missing.csv", "target")
