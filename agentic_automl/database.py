"""SQLite storage for experiments; nested results remain JSON payloads."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent.parent / "results" / "history.sqlite3"


@contextmanager
def connection(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a short transaction, then commit/rollback and always close it."""
    path = path if path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path, timeout=5)
    try:
        database.execute("PRAGMA foreign_keys = ON")
        database.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY NOT NULL,
                completed_at TEXT,
                dataset TEXT,
                task TEXT CHECK (task IN ('classification', 'regression')),
                parent_id TEXT REFERENCES experiments(id) DEFERRABLE INITIALLY DEFERRED,
                payload TEXT NOT NULL
            )
        """)
        database.execute(
            "CREATE INDEX IF NOT EXISTS experiments_date ON experiments(completed_at)"
        )
        with database:
            yield database
    finally:
        database.close()


def insert_experiment(
    database: sqlite3.Connection,
    record: dict,
    *,
    skip_existing: bool = False,
) -> bool:
    """Insert within the caller's transaction; never overwrite an existing run."""
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("id"), str)
        or not record["id"]
    ):
        raise ValueError("Each experiment must have a non-empty string id.")
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    conflict = " ON CONFLICT(id) DO NOTHING" if skip_existing else ""
    cursor = database.execute(
        "INSERT INTO experiments (id, completed_at, dataset, task, parent_id, payload) "
        "VALUES (?, ?, ?, ?, ?, ?)" + conflict,
        (
            record["id"],
            record.get("completed_at"),
            record.get("dataset"),
            record.get("task"),
            record.get("parent_id"),
            payload,
        ),
    )
    return cursor.rowcount == 1


def save_experiment(record: dict) -> None:
    with connection() as database:
        insert_experiment(database, record)


def load_experiments() -> list[dict]:
    with connection() as database:
        rows = database.execute(
            "SELECT payload FROM experiments ORDER BY completed_at DESC, rowid DESC"
        ).fetchall()
    return [json.loads(row[0]) for row in rows]
