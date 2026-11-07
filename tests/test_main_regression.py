from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_automl.regression.comparison import compare_regression_models
from main_regression import MAX_SAMPLES, limit_samples, main, print_results


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
    assert tables["example.csv"].columns.tolist() == [
        "cv_mse_mean",
        "cv_mse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "test_mse",
        "test_r2",
    ]
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


def test_display_shows_cv_means_and_only_winner_test_scores(capsys) -> None:
    table = pd.DataFrame(
        {
            "cv_mse_mean": [1, 2],
            "cv_mse_std": [0.1, 0.2],
            "cv_r2_mean": [0.8, 0.7],
            "cv_r2_std": [0.01, 0.02],
            "test_mse": [1.1, float("nan")],
            "test_r2": [0.75, float("nan")],
        },
        index=["winner", "other"],
    )
    table.attrs.update(selected_model="winner", sample_rows=100, rejected_columns={})
    original = table.copy()
    print_results({"example.csv": table})
    output = capsys.readouterr().out
    assert "CV averages:" in output and "Final test results:" in output
    assert "std" not in output and "nan" not in output.lower()
    assert "other" in output.split("Final test results:")[0]
    assert "other" not in output.split("Final test results:")[1]
    pd.testing.assert_frame_equal(table, original)


def test_runner_applies_regression_model_selection_and_search(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("main_regression.DATASETS", {"example.csv": "price"})
    pd.DataFrame({"value": range(40), "price": range(40)}).to_csv(
        tmp_path / "example.csv", index=False
    )
    table = main(
        data_dir=tmp_path,
        model_names=["ridge", "random_forest"],
        random_search_trials=1,
    )["example.csv"]
    assert table.index.tolist() == [
        "mean_baseline",
        "ridge",
        "random_forest",
        "ridge_random_search",
        "random_forest_random_search",
    ]
    assert set(table.attrs["random_search"]) == {"ridge", "random_forest"}
    assert table.filter(regex="^test_").notna().any(axis=1).sum() == 1
