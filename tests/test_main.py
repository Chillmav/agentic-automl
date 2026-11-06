import pandas as pd
import pytest

from main import MAX_SAMPLES, limit_samples, main


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
    monkeypatch.setattr("main.DATASETS", {"airlines.csv": None})
    main()
    assert "Skipping airlines.csv" in capsys.readouterr().out
