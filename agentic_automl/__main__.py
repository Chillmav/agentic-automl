"""Run with python -m agentic_automl."""

import argparse
import json
from pathlib import Path
from agentic_automl.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible binary classification experiment."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--positive-class", required=True)
    parser.add_argument("--output", required=True, type=Path, help="A new artifact directory")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    try:
        report = run_experiment(
            args.csv, args.target, args.positive_class, args.output, args.random_state
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
