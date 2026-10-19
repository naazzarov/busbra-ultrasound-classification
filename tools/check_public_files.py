#!/usr/bin/env python3
"""Fail if tracked/staged content includes private data or generated artifacts."""
from pathlib import Path
import re
import subprocess
import sys

PRIVATE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".parquet", ".arrow", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pdf", ".rar", ".zip"}
PRIVATE_PARTS = {"private_handoff", "private_outputs", "source_data", "ready_dataset", "comparison_output", ".env"}
PRIVATE_NAMES = {"run_info.json", "internal_run.json"}
ROW_IDENTIFIER = re.compile(rb"(?m)^(?:bus_\d{4}-[lrs]|case_\d{4})[,\t]")


def check():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = []
    for name in filter(None, paths):
        path = Path(name)
        if path.suffix.lower() in PRIVATE_SUFFIXES or set(path.parts) & PRIVATE_PARTS or path.name in PRIVATE_NAMES:
            failures.append(name)
            continue
        # Inspect index contents: these are the exact bytes a commit will publish.
        raw = subprocess.check_output(["git", "show", f":{name}"])
        if ROW_IDENTIFIER.search(raw):
            failures.append(name)
    if failures:
        print("Private artifact publication blocked: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"Privacy check passed for {len(list(filter(None, paths)))} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
