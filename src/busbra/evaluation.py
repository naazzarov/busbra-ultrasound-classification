"""Nested evaluation: scaling and fitting stay inside patient-grouped folds."""
from dataclasses import dataclass
from importlib.metadata import version
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import ValidatedDataset
from .protocol import read_contract, svm_grid


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    selected_features: pd.DataFrame
    folds: list[dict]
    metadata: dict


def classification_metrics(truth, predicted) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    if tn + fp == 0 or tp + fn == 0:
        raise ValueError("Both classes are required to compute sensitivity and specificity")
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "malignant_sensitivity": float(tp / (tp + fn)),
        "benign_specificity": float(tn / (tn + fp)),
    }


def evaluate_baseline(dataset: ValidatedDataset, *, n_jobs: int = 2) -> EvaluationResult:
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    table = dataset.table
    names = list(dataset.feature_names)
    features = table[names].to_numpy(dtype=float)
    target = table.diagnosis.to_numpy(dtype=int)
    groups = table.patient_id.to_numpy()
    contract = read_contract()
    held_out_count = np.zeros(len(table), dtype=int)
    prediction = np.full(len(table), -1, dtype=int)
    fold_results = []
    selected = []
    for fold in contract["outer_folds"]:
        train = table.fold5_kfold.to_numpy() != fold
        test = ~train
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Patient leakage between outer training and test partitions")
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold)
        inner = list(splitter.split(features[train], target[train], groups[train]))
        for fit, valid in inner:
            if set(groups[train][fit]) & set(groups[train][valid]):
                raise ValueError("Patient leakage in inner cross-validation")
            if set(target[train][fit]) != {0, 1} or set(target[train][valid]) != {0, 1}:
                raise ValueError("Inner training and validation partitions need both classes")
        model = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
            svm_grid(), scoring="f1_macro", cv=inner, refit=True,
            n_jobs=n_jobs, pre_dispatch=n_jobs, error_score="raise",
        )
        started = perf_counter()
        model.fit(features[train], target[train])
        search_and_refit_seconds = perf_counter() - started
        started = perf_counter()
        prediction[test] = model.predict(features[test])
        prediction_seconds = perf_counter() - started
        held_out_count[test] += 1
        fold_results.append({
            "outer_fold": fold, "inner_seed": 42 + fold,
            "training_rows": int(train.sum()), "test_rows": int(test.sum()),
            "n_candidates": len(model.cv_results_["params"]),
            "best_params": model.best_params_,
            "inner_f1_macro": float(model.best_score_),
            "search_and_refit_seconds": search_and_refit_seconds,
            "prediction_seconds": prediction_seconds,
            "selected_feature_count": len(names), "feature_reduction": 0.0,
            "metrics": classification_metrics(target[test], prediction[test]),
        })
        selected.extend({"outer_fold": fold, "feature_name": name} for name in names)
    if not np.all(held_out_count == 1) or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Every image must have exactly one valid held-out prediction")
    predictions = pd.DataFrame({
        "image_id": table.image_id.to_numpy(), "patient_id": groups,
        "outer_fold": table.fold5_kfold.to_numpy(), "y_true": target, "y_pred": prediction,
    })
    metadata = {
        "protocol_id": contract["protocol_id"], "method": "baseline",
        "dataset_sha256": dataset.sha256, "run_status": "completed",
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "n_jobs": n_jobs,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "summary": {
            name: {"mean": float(np.mean([f["metrics"][name] for f in fold_results])),
                   "sample_sd": float(np.std([f["metrics"][name] for f in fold_results], ddof=1))}
            for name in fold_results[0]["metrics"]
        },
        "format_note": "Internal run record. Original ZIP templates are required for official submission export.",
    }
    return EvaluationResult(predictions, pd.DataFrame(selected), fold_results, metadata)
