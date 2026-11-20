import pytest


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch) -> None:
    """All tests use temporary files and a temporary database."""
    monkeypatch.setattr(
        "agentic_automl.database.DATABASE_PATH", tmp_path / "history.sqlite3"
    )
    monkeypatch.setattr(
        "agentic_automl.history.HISTORY_FILE", tmp_path / "history.jsonl"
    )
