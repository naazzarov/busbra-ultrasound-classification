"""Export exact original handoff columns, plus full private audit records."""
import json
from pathlib import Path
import shutil

import pandas as pd


def export_submission(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result.predictions.to_csv(output / "predictions.csv", index=False)
    result.selected_features.to_csv(output / "selected_features.csv", index=False)
    params, timings = [], []
    for fold in result.folds:
        p = fold["best_params"]
        params.append({"outer_fold": fold["outer_fold"], "kernel": p["svm__kernel"], "C": p["svm__C"],
                       "gamma": p.get("svm__gamma", "scale"), "class_weight": p["svm__class_weight"],
                       "inner_macro_f1": fold["inner_f1_macro"], "inner_accuracy": fold["inner_accuracy"]})
        timings.append({"outer_fold": fold["outer_fold"], "selector_seconds": fold["selector_seconds"],
                        "svm_tuning_seconds": fold["svm_tuning_seconds"], "refit_seconds": fold["refit_seconds"],
                        "predict_seconds": fold["prediction_seconds"], "total_seconds": fold["total_seconds"]})
    pd.DataFrame(params).to_csv(output / "best_params.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    (output / "run_info.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + "\n")
    (output / "internal_run.json").write_text(json.dumps({**result.metadata, "folds": result.folds}, indent=2, allow_nan=False) + "\n")
    shutil.copytree(Path(__file__).parent, output / "code" / "busbra", ignore=shutil.ignore_patterns("__pycache__"))
    root = Path(__file__).resolve().parents[2]
    for name in ["METHODS.md", "REFERENCE_AUDIT.md"]:
        if (root / "docs" / name).exists():
            shutil.copyfile(root / "docs" / name, output / name)
    (output / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in result.metadata["versions"].items()) + "\n")
    method = result.metadata["method"]
    if result.metadata.get("protocol_id") == "BUSBRA-ROBUSTNESS-20261028-V1":
        shutil.copyfile(root / "tools" / "run_robustness.py", output / "code" / "run_robustness.py")
        command = f"--data PATH/TO/UNCHANGED.csv --experiments {method} --output NEW_PRIVATE_DIR --jobs 2"
        (output / "README.md").write_text(
            f"# Private robustness experiment: {method}\n\nKeep all contents private.\n"
            f"Repository: PYTHONPATH=src python tools/run_robustness.py {command}\n"
            f"Standalone: PYTHONPATH=code python code/run_robustness.py {command}\n"
            "Install requirements.txt using the recorded Python version. See internal_run.json.\n")
        return
    if result.metadata.get("protocol_id") == "BUSBRA-SECONDARY-20261028-V1":
        shutil.copyfile(root / "tools" / "run_secondary.py", output / "code" / "run_secondary.py")
        data_arg = " --data PATH/TO/UNCHANGED.csv" if result.metadata["dataset"] == "busbra" else ""
        author_arg = " --author-code PATH/TO/PINNED_PRIVATE_GBFRS.py" if method == "author_gbfrs" else ""
        command = (f"--dataset {result.metadata['dataset']} --methods {method}"
                   f"{data_arg}{author_arg} --output NEW_PRIVATE_DIR --jobs 2")
        (output / "README.md").write_text(
            f"# Private secondary {method} submission\n\n"
            "Do not upload this folder to GitHub. Install requirements.txt with the recorded Python version.\n"
            f"Repository: PYTHONPATH=src python tools/run_secondary.py {command}\n"
            f"Standalone: PYTHONPATH=code python code/run_secondary.py {command}\n"
            "Author code is never bundled. See internal_run.json for the full nested search audit.\n")
        return
    (output / "README.md").write_text(
        f"# Private {method} submission\n\n"
        "Do not upload this folder to GitHub. Install requirements.txt with the recorded Python version.\n"
        f"Reproduce from the repository: busbra {method} --data PATH/TO/UNCHANGED.csv --output NEW_PRIVATE_DIR --jobs 2\n"
        f"Standalone: PYTHONPATH=code python -m busbra.cli {method} --data PATH/TO/UNCHANGED.csv --output NEW_PRIVATE_DIR --jobs 2\n"
        "See run_info.json for versions, fixed selector settings and timing semantics; internal_run.json contains all inner scores.\n")
