"""Command-line validation and three-method experiments."""
import argparse
import json
from pathlib import Path
import sys

from .data import load_dataset
from .evaluation import evaluate_baseline, evaluate_method
from .analysis import summarize
from datetime import datetime, timezone
from .export import export_submission
from .protocol import read_contract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BUS-BRA frozen-dataset experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["validate", "baseline", "fuzzy_rough", "gbfrs", "run-all", "summarize"]:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--data", type=Path, default=Path(read_contract()["dataset_filename"]))
        if command != "validate":
            command_parser.add_argument("--output", type=Path, required=True, help="New private output directory")
            command_parser.add_argument("--jobs", type=int, default=2)
        if command == "summarize":
            command_parser.add_argument("--submissions", type=Path, required=True)
            command_parser.add_argument("--bootstrap", type=int, default=5000)
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
        if args.command == "summarize":
            result = summarize(dataset, args.submissions, n_bootstrap=args.bootstrap)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"Aggregate-only research summary: {args.output}")
            return 0
        def progress(method, fold, count, seconds):
            print(f"{method}: completed outer fold {fold}/5, {count} features, {seconds:.1f}s", flush=True)
        if args.command == "run-all":
            args.output.mkdir(parents=True, exist_ok=False)
            manifest = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                        "execution": "methods run sequentially; native threads limited to one",
                        "protocol_id": read_contract()["protocol_id"], "dataset_sha256": dataset.sha256, "completed_methods": []}
            record_path = args.output / "experiment.json"
            try:
                for method in ("baseline", "fuzzy_rough", "gbfrs"):
                    record_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    print(f"Starting {method}: full frozen nested evaluation", flush=True)
                    result = evaluate_method(dataset, method=method, n_jobs=args.jobs, progress=progress)
                    export_submission(result, args.output / "submissions" / method)
                    manifest["completed_methods"].append(method)
                manifest["status"] = "COMPLETED"
                manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            except Exception:
                manifest["status"] = "FAILED"
                raise
            finally:
                record_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"All three complete private submissions: {args.output}")
            return 0
        result = evaluate_method(dataset, method=args.command, n_jobs=args.jobs, progress=progress)
        export_submission(result, args.output)
        print(f"Completed five {args.command} folds. Private artifacts: {args.output}")
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
