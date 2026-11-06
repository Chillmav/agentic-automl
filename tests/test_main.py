import pandas as pd
import pytest

from main_classification import MAX_SAMPLES, limit_samples, main, print_results


@pytest.mark.parametrize("rows", [100, MAX_SAMPLES, MAX_SAMPLES + 1000])
def test_sample_limit_alignment_and_reproducibility(rows: int) -> None:
    features = pd.DataFrame({"value": range(rows)})
    target = pd.Series([0] * (rows * 9 // 10) + [1] * (rows // 10))
    sampled_x, sampled_y = limit_samples(features, target)
    assert len(sampled_x) == min(rows, MAX_SAMPLES)
    assert sampled_x.index.is_unique
    pd.testing.assert_series_equal(sampled_y, target.loc[sampled_x.index])
    assert sampled_y.mean() == pytest.approx(0.1)
    repeated_x, repeated_y = limit_samples(features, target)
    pd.testing.assert_frame_equal(sampled_x, repeated_x)
    pd.testing.assert_series_equal(sampled_y, repeated_y)


def test_runner_skips_datasets_without_configured_target(monkeypatch, capsys) -> None:
    monkeypatch.setattr("main_classification.DATASETS", {"airlines.csv": None})
    assert main() == {}
    assert capsys.readouterr().out == ""


def test_display_hides_stds_and_untested_models_test_scores(capsys) -> None:
    table = pd.DataFrame(
        {
            "cv_f1_mean": [0.8, 0.7],
            "cv_f1_std": [0.1, 0.2],
            "test_f1": [0.9, float("nan")],
        },
        index=["winner", "other_model"],
    )
    table.attrs["selected_model"] = "winner"
    original = table.copy()
    print_results({"example.csv": table})
    output = capsys.readouterr().out
    assert "CV averages:" in output
    assert "Final test results:" in output
    assert "std" not in output
    assert "nan" not in output.lower()
    assert "other_model" in output.split("Final test results:")[0]
    assert "other_model" not in output.split("Final test results:")[1]
    assert "0.8" in output and "0.9" in output
    pd.testing.assert_frame_equal(table, original)


def test_runner_returns_tables_without_printing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("main_classification.DATASETS", {"example.csv": "target"})
    pd.DataFrame({"signal": [0, 1] * 20, "target": [0, 1] * 20}).to_csv(
        tmp_path / "example.csv", index=False
    )
    tables = main(data_dir=tmp_path)
    table = tables["example.csv"]
    assert isinstance(table, pd.DataFrame)
    assert table.shape == (7, 12)
    assert table.attrs == {
        "positive_class": "1",
        "selected_model": "logistic_regression",
        "selection_metric": "cv_f1_mean",
        "sample_rows": 40,
        "duplicate_rows": 38,
    }
    assert capsys.readouterr().out == ""
