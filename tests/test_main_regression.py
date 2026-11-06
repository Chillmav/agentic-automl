from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_automl.regression.comparison import compare_regression_models
from main_regression import MAX_SAMPLES, limit_samples, main


@pytest.mark.parametrize("rows", [100, MAX_SAMPLES + 1000])
def test_regression_sampling_is_capped_aligned_and_reproducible(rows: int) -> None:
    features, target = pd.DataFrame({"value": range(rows)}), pd.Series(range(rows))
    sampled_x, sampled_y = limit_samples(features, target)
    assert len(sampled_x) == min(rows, MAX_SAMPLES)
    pd.testing.assert_series_equal(sampled_y, target.loc[sampled_x.index])
    pd.testing.assert_frame_equal(sampled_x, limit_samples(features, target)[0])


def test_runner_returns_regression_tables(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("main_regression.DATASETS", {"example.csv": "price"})
    pd.DataFrame({"size": range(100), "price": range(100)}).to_csv(
        tmp_path / "example.csv", index=False
    )
    tables = main(data_dir=tmp_path)
    assert tables["example.csv"].columns.tolist() == ["mse", "r2"]
    assert tables["example.csv"].attrs["sample_rows"] == 100
    assert tables["example.csv"].attrs["rejected_columns"] == {}
    assert capsys.readouterr().out == ""


def test_walmart_split_keeps_entire_dates_together(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "main_regression.DATASETS", {"Walmart_Sales.csv": "Weekly_Sales"}
    )
    pd.DataFrame(
        {
            "Date": [f"{day:02d}-01-2020" for day in range(1, 8) for _ in range(3)],
            "Store": [1, 2, 3] * 7,
            "Holiday_Flag": [0] * 21,
            "Weekly_Sales": range(21),
        }
    ).to_csv(tmp_path / "Walmart_Sales.csv", index=False)
    with patch(
        "main_regression.compare_regression_models", wraps=compare_regression_models
    ) as compare:
        tables = main(data_dir=tmp_path)
    assert compare.call_args.kwargs["shuffle"] is False
    assert compare.call_args.kwargs["test_size"] == 6
    assert tables["Walmart_Sales.csv"].attrs["split_strategy"] == "chronological"
