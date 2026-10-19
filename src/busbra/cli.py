"""Command-line validation and three-method experiments."""
import argparse
import json
from pathlib import Path
import sys

from .data import load_dataset
from .evaluation import evaluate_baseline, evaluate_method
from .export import export_submission
from .protocol import read_contract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BUS-BRA frozen-dataset experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["validate", "baseline", "fuzzy_rough", "gbfrs"]:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--data", type=Path, default=Path(read_contract()["dataset_filename"]))
        if command != "validate":
            command_parser.add_argument("--output", type=Path, required=True, help="New private output directory")
            command_parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.data)
        if args.command == "validate":
            print(json.dumps({"status": "valid", "rows": len(dataset.table), "predictors": len(dataset.feature_names), "sha256": dataset.sha256}, indent=2))
            return 0
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        if args.output.exists():
            raise ValueError("Output path already exists; choose a new directory to preserve prior results")
        def progress(method, fold, count, seconds):
            print(f"{method}: completed outer fold {fold}/5, {count} features, {seconds:.1f}s", flush=True)
        result = evaluate_method(dataset, method=args.command, n_jobs=args.jobs, progress=progress)
        export_submission(result, args.output)
        print(f"Completed five {args.command} folds. Private artifacts: {args.output}")
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
