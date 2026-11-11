"""Small local history: one completed experiment per JSON line."""

import json
from pathlib import Path
from threading import Lock

HISTORY_FILE = Path(__file__).resolve().parent.parent / "results" / "experiments.jsonl"
_lock = Lock()


def load_history() -> list[dict]:
    """Return newest entries first; a missing file means no experiments yet."""
    with _lock:
        if not HISTORY_FILE.exists():
            return []
        with HISTORY_FILE.open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
    return list(reversed(records))


def save_experiment(record: dict) -> None:
    """Append a result without rewriting earlier experiments."""
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with _lock:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8") as file:
            file.write(line)
