"""Nested, patient-grouped evaluation shared by all three methods."""
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
from pathlib import Path
import platform
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import ValidatedDataset
from .protocol import read_contract, svm_grid
from .selectors import make_selector


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


def choose_candidate(results):
    """Exact ties: macro-F1, accuracy, fewer features, original grid order."""
    return max(range(len(results)), key=lambda i: (
        results[i]["macro_f1"], results[i]["accuracy"], -results[i]["n_features"], -i))


def _prepare(X, y, method):
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    selector = make_selector(method)
    start = perf_counter()
    if selector is not None:
        scaled = selector.fit_transform(scaled, y)
        indices = selector.get_support(indices=True)
        audit = {"dependency_history": selector.dependency_history_}
        if method == "gbfrs":
            audit.update(n_balls=len(selector.ball_sizes_), ball_sample_count=int(selector.ball_sizes_.sum()))
    else:
        indices, audit = np.arange(X.shape[1]), {}
    return scaler, selector, scaled, indices, perf_counter() - start if selector is not None else 0.0, audit


def _transform(scaler, selector, X):
    scaled = scaler.transform(X)
    return selector.transform(scaled) if selector is not None else scaled


def _score_candidate(params, prepared):
    f1, accuracy, counts = [], [], []
    params = {key.removeprefix("svm__"): value for key, value in params.items()}
    for train_X, train_y, valid_X, valid_y in prepared:
        prediction = SVC(**params).fit(train_X, train_y).predict(valid_X)
        f1.append(f1_score(valid_y, prediction, labels=[0, 1], average="macro", zero_division=0))
        accuracy.append(accuracy_score(valid_y, prediction))
        counts.append(train_X.shape[1])
    return {"macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(accuracy)),
            "n_features": float(np.mean(counts)), "split_macro_f1": f1, "split_accuracy": accuracy}


def evaluate_method(dataset: ValidatedDataset, *, method="baseline", n_jobs=2, progress=None) -> EvaluationResult:
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    make_selector(method)  # Validate before doing work.
    table = dataset.table
    names = list(dataset.feature_names)
    features = table[names].to_numpy(dtype=float)
    target = table.diagnosis.to_numpy(dtype=int)
    groups = table.patient_id.to_numpy()
    contract = read_contract()
    grid = list(ParameterGrid(svm_grid()))
    held_out_count = np.zeros(len(table), dtype=int)
    prediction = np.full(len(table), -1, dtype=int)
    fold_results, selected = [], []
    # Limit native threads once, outside shared-memory candidate workers.
    with threadpool_limits(limits=1):
        for fold in contract["outer_folds"]:
            fold_start = perf_counter()
            train = table.fold5_kfold.to_numpy() != fold
            test = ~train
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Patient leakage between outer training and test partitions")
            X, y, g = features[train], target[train], groups[train]
            inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold).split(X, y, g))
            prepared, inner_audit, selector_seconds = [], [], 0.0
            for fit, valid in inner:
                if set(g[fit]) & set(g[valid]):
                    raise ValueError("Patient leakage in inner cross-validation")
                if set(y[fit]) != {0, 1} or set(y[valid]) != {0, 1}:
                    raise ValueError("Inner training and validation partitions need both classes")
                scaler, selector, transformed, indices, elapsed, audit = _prepare(X[fit], y[fit], method)
                selector_seconds += elapsed
                prepared.append((transformed, y[fit], _transform(scaler, selector, X[valid]), y[valid]))
                inner_audit.append({"training_rows": len(fit), "validation_rows": len(valid),
                                    "selected_indices": indices.tolist(), **audit})
            started = perf_counter()
            scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_score_candidate)(params, prepared) for params in grid)
            tuning_seconds = perf_counter() - started
            best = choose_candidate(scores)
            started = perf_counter()
            scaler, selector, transformed, indices, elapsed, audit = _prepare(X, y, method)
            selector_seconds += elapsed
            params = {key.removeprefix("svm__"): value for key, value in grid[best].items()}
            model = SVC(**params).fit(transformed, y)
            refit_seconds = perf_counter() - started
            started = perf_counter()
            prediction[test] = model.predict(_transform(scaler, selector, features[test]))
            predict_seconds = perf_counter() - started
            held_out_count[test] += 1
            result = {
                "outer_fold": fold, "inner_seed": 42 + fold,
                "training_rows": int(train.sum()), "test_rows": int(test.sum()),
                "n_candidates": len(grid), "best_params": grid[best],
                "inner_f1_macro": scores[best]["macro_f1"], "inner_accuracy": scores[best]["accuracy"],
                "selector_seconds": selector_seconds, "svm_tuning_seconds": tuning_seconds,
                "refit_seconds": refit_seconds, "outer_selector_seconds": elapsed,
                "prediction_seconds": predict_seconds, "total_seconds": perf_counter() - fold_start,
                "search_and_refit_seconds": tuning_seconds + refit_seconds,
                "selected_feature_count": len(indices), "feature_reduction": 1 - len(indices) / len(names),
                "inner_partitions": inner_audit, "selector_audit": audit,
                "search_results": [{"params": p, **s} for p, s in zip(grid, scores)],
                "metrics": classification_metrics(target[test], prediction[test]),
            }
            fold_results.append(result)
            selected.extend({"outer_fold": fold, "feature_name": names[i]} for i in indices)
            if progress:
                progress(method, fold, len(indices), result["total_seconds"])
    if not np.all(held_out_count == 1) or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Every image must have exactly one valid held-out prediction")
    predictions = pd.DataFrame({
        "image_id": table.image_id.to_numpy(), "patient_id": groups,
        "outer_fold": table.fold5_kfold.to_numpy(), "y_true": target, "y_pred": prediction,
    })
    metadata = {
        "protocol_id": contract["protocol_id"], "method": method,
        "dataset_sha256": dataset.sha256, "status": "COMPLETED", "run_status": "completed",
        "seed": 42, "outer_folds": contract["outer_folds"], "scoring": "f1_macro",
        "inner_cv": "StratifiedGroupKFold(3,shuffle=True,random_state=42+outer_fold)",
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "n_jobs": n_jobs, "native_threads": 1,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "implementation_note": "Independent paper reconstruction; see docs/METHODS.md. GBFRS is not byte-for-byte author code.",
        "selector_settings": {"baseline": {}, "fuzzy_rough": {"tol": 1e-12},
                              "gbfrs": {"purity": 0.9, "random_state": 42, "tol": 1e-12}}[method],
        "timing_note": "selector_seconds includes 3 inner fits and the outer fit. refit_seconds includes the outer selector; do not sum these overlapping columns. total_seconds is wall time.",
        "summary": {
            name: {"mean": float(np.mean([f["metrics"][name] for f in fold_results])),
                   "sample_sd": float(np.std([f["metrics"][name] for f in fold_results], ddof=1))}
            for name in fold_results[0]["metrics"]
        },
    }
    return EvaluationResult(predictions, pd.DataFrame(selected), fold_results, metadata)


def evaluate_baseline(dataset, *, n_jobs=2):
    return evaluate_method(dataset, method="baseline", n_jobs=n_jobs)
