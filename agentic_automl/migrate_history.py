"""One-time experiment import; the JSONL source is kept untouched."""

import argparse
import json
import sqlite3
from pathlib import Path

from agentic_automl import database
from agentic_automl.history import HISTORY_FILE


def migrate_experiments(source: Path, destination: Path) -> int:
    """Import atomically; rerunning skips IDs already present in SQLite."""
    imported = 0
    with (
        source.open(encoding="utf-8") as file,
        database.connection(destination) as connection,
    ):
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                imported += database.insert_experiment(
                    connection, record, skip_existing=True
                )
            except (ValueError, sqlite3.Error) as error:
                raise ValueError(f"{source}, line {number}: {error}") from error
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HISTORY_FILE)
    parser.add_argument("--database", type=Path, default=database.DATABASE_PATH)
    arguments = parser.parse_args()
    try:
        count = migrate_experiments(arguments.source, arguments.database)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    print(
        f"Imported {count} experiments into {arguments.database}. Source file preserved."
    )


if __name__ == "__main__":
    main()
