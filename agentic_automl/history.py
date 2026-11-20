"""Experiments use SQLite; analyst reports and decisions still use JSONL."""

import json
from pathlib import Path
from threading import Lock

from agentic_automl import database

# Legacy source for migration; reports still live alongside this file.
HISTORY_FILE = Path(__file__).resolve().parent.parent / "results" / "experiments.jsonl"
_lock = Lock()


def _load_records(path: Path) -> list[dict]:
    with _lock:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
    return list(reversed(records))


def _append_record(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line)


def load_history() -> list[dict]:
    """Return newest experiments first, reading from disk."""
    return database.load_experiments()


def save_experiment(record: dict) -> None:
    database.save_experiment(record)


def load_analyses(experiment_id: str) -> list[dict]:
    records = _load_records(HISTORY_FILE.with_name("analyses.jsonl"))
    decisions = _load_records(HISTORY_FILE.with_name("analysis_decisions.jsonl"))
    rejected = {
        item["analysis_id"]
        for item in decisions
        if item["experiment_id"] == experiment_id
    }
    return [
        {**record, "rejected": record["id"] in rejected}
        for record in records
        if record["experiment_id"] == experiment_id
    ]


def save_analysis(record: dict) -> None:
    """Preserve every report without modifying experiment records."""
    _append_record(HISTORY_FILE.with_name("analyses.jsonl"), record)


def save_rejection(record: dict) -> None:
    _append_record(HISTORY_FILE.with_name("analysis_decisions.jsonl"), record)
