#!/usr/bin/env python3
"""Fail if tracked/staged content includes private data or generated artifacts."""
from pathlib import Path
import re
import subprocess
import sys

PRIVATE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".parquet", ".arrow", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pdf", ".rar", ".zip", ".pt", ".pth", ".onnx", ".h5", ".hdf5", ".dcm", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PRIVATE_PARTS = {"private_handoff", "private_outputs", "source_data", "ready_dataset", "comparison_output", ".env"}
PRIVATE_NAMES = {"run_info.json", "internal_run.json"}
ROW_IDENTIFIER = re.compile(rb"\b(?:bus_\d{4}-[lrs]|case_\d{4})\b")
PUBLIC_FIGURES = {"paper/figures/performance-reduction.png", "paper/figures/paired-effects.png", "paper/figures/label-noise.png"}


def check():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = []
    for name in filter(None, paths):
        path = Path(name)
        if (path.suffix.lower() in PRIVATE_SUFFIXES or (path.suffix.lower() == ".png" and name not in PUBLIC_FIGURES)
                or set(path.parts) & PRIVATE_PARTS or path.name in PRIVATE_NAMES | {"experiment.json", "completion_manifest.json"}):
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
