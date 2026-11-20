import json
import sqlite3

import pytest

from agentic_automl import database
from agentic_automl.migrate_history import migrate_experiments


def test_database_preserves_nested_results_and_orders_by_date() -> None:
    latest = {
        "id": "latest",
        "dataset": "données.csv",
        "task": "regression",
        "completed_at": "2026-11-20T10:00:00+00:00",
        "cv_results": [{"model": "ridge", "mse": 1.25}],
        "training_profile": {"columns": [{"name": "age", "missing_percent": 5.0}]},
    }
    older = {"id": "older", "completed_at": "2026-11-19T10:00:00+00:00"}
    database.save_experiment(latest)
    database.save_experiment(older)
    assert database.load_experiments() == [latest, older]


def test_duplicate_id_cannot_overwrite_result() -> None:
    original = {"id": "run", "dataset": "original.csv"}
    database.save_experiment(original)
    with pytest.raises(sqlite3.IntegrityError):
        database.save_experiment({"id": "run", "dataset": "changed.csv"})
    assert database.load_experiments() == [original]


def test_follow_up_requires_existing_parent() -> None:
    with pytest.raises(sqlite3.IntegrityError):
        database.save_experiment({"id": "child", "parent_id": "missing"})
    assert database.load_experiments() == []
    parent = {"id": "parent"}
    child = {"id": "child", "parent_id": "parent"}
    database.save_experiment(parent)
    database.save_experiment(child)
    assert database.load_experiments() == [child, parent]


def test_user_text_is_stored_as_data() -> None:
    record = {"id": "run'); DROP TABLE experiments;--", "dataset": "quote's.csv"}
    database.save_experiment(record)
    assert database.load_experiments() == [record]


def test_migration_is_repeatable_and_preserves_source_and_ids(tmp_path) -> None:
    source = tmp_path / "experiments.jsonl"
    destination = tmp_path / "history.sqlite3"
    records = [
        {
            "id": "salary",
            "dataset": "Salary_dataset.csv",
            "test_results": [{"r2": 0.96}],
        },
        {
            "id": "advertising",
            "dataset": "Advertising Budget and Sales.xls",
            "rejected_columns": {},
        },
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    original = source.read_bytes()
    assert migrate_experiments(source, destination) == 2
    assert migrate_experiments(source, destination) == 0
    assert database.load_experiments() == list(reversed(records))
    assert source.read_bytes() == original


def test_migration_failure_rolls_back_every_new_record(tmp_path) -> None:
    database.save_experiment({"id": "existing"})
    source = tmp_path / "experiments.jsonl"
    source.write_text('{"id":"new"}\ninvalid JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        migrate_experiments(source, database.DATABASE_PATH)
    assert database.load_experiments() == [{"id": "existing"}]
    assert source.read_text(encoding="utf-8").endswith("invalid JSON\n")


def test_migration_supports_children_before_parents(tmp_path) -> None:
    source = tmp_path / "experiments.jsonl"
    source.write_text(
        '{"id":"child","parent_id":"parent"}\n{"id":"parent"}\n', encoding="utf-8"
    )
    assert migrate_experiments(source, database.DATABASE_PATH) == 2
    assert {record["id"] for record in database.load_experiments()} == {
        "parent",
        "child",
    }


def test_migration_does_not_overwrite_existing_id(tmp_path) -> None:
    database.save_experiment({"id": "run", "dataset": "preserved.csv"})
    source = tmp_path / "experiments.jsonl"
    source.write_text('{"id":"run","dataset":"different.csv"}\n', encoding="utf-8")
    assert migrate_experiments(source, database.DATABASE_PATH) == 0
    assert database.load_experiments() == [{"id": "run", "dataset": "preserved.csv"}]
