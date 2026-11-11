from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic_automl import history


def test_history_persists_newest_first(tmp_path, monkeypatch) -> None:
    path = tmp_path / "results" / "experiments.jsonl"
    monkeypatch.setattr(history, "HISTORY_FILE", path)
    assert history.load_history() == []
    history.save_experiment({"id": "first", "dataset": "données.csv"})
    history.save_experiment({"id": "second"})
    # Reading again uses disk, not an in-memory cache.
    assert history.load_history() == [
        {"id": "second"},
        {"id": "first", "dataset": "données.csv"},
    ]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_concurrent_appends_keep_every_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "history.jsonl")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(history.save_experiment, ({"id": i} for i in range(20))))
    assert {record["id"] for record in history.load_history()} == set(range(20))


def test_invalid_record_does_not_change_existing_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "history.jsonl")
    history.save_experiment({"id": "first"})
    with pytest.raises(ValueError):
        history.save_experiment({"score": float("nan")})
    assert history.load_history() == [{"id": "first"}]
