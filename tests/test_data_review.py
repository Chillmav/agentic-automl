import pandas as pd
import pytest

from agentic_automl.data_review import apply_changes, review_dataset


def test_date_expansion_and_missing_targets_preserve_source() -> None:
    data = pd.DataFrame(
        {"date": ["2024-02-29", None, "2024-03-01"], "target": [1.0, 2.0, None]}
    )
    original = data.copy(deep=True)
    report = review_dataset(data, "target")
    prepared = apply_changes(data, report["proposed_changes"])
    assert len(prepared) == 2
    assert "date" not in prepared
    assert prepared.loc[0, "date__year"] == 2024
    assert prepared.loc[0, "date__month"] == 2
    assert prepared.loc[0, "date__day"] == 29
    assert prepared.loc[0, "date__weekday"] == 3
    assert pd.isna(prepared.loc[1, "date__year"])
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(apply_changes(data, []), original)


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-02-30", "2024-03-01"],
        ["2024-03-01", "01/02/2024"],
        ["01/02/2024", "02/03/2024"],
    ],
)
def test_uncertain_dates_are_not_converted(dates) -> None:
    report = review_dataset(pd.DataFrame({"date": dates, "y": [1, 2]}), "y")
    assert report["proposed_changes"] == []


def test_generated_names_cannot_overwrite_existing_columns() -> None:
    data = pd.DataFrame({"date": ["2024-01-01"], "date__year": [7], "y": [1]})
    report = review_dataset(data, "y")
    assert report["proposed_changes"] == []
    assert any("already exist" in warning for warning in report["warnings"])


def test_target_is_never_expanded_as_date() -> None:
    data = pd.DataFrame({"x": [1, 2], "y": ["2024-01-01", "2024-01-02"]})
    assert review_dataset(data, "y")["proposed_changes"] == []
