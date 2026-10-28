"""Publication aggregates from validated out-of-fold predictions; no row export.

Intervals resample patients and condition on the fitted OOF predictions. They do
not estimate the uncertainty of retraining the entire learning procedure.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import classification_metrics
from .protocol import read_contract

METHODS = ("baseline", "fuzzy_rough", "gbfrs")


def metrics_from_counts(counts):
    tn, fp, fn, tp = np.asarray(counts, dtype=float)
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return (negative_f1 + positive_f1) / 2


def selection_stability(sets, universe):
    """Named-feature stability across overlapping outer training sets.

    Nogueira et al. (2018), Definition 4 / Eq. 2; reported descriptively, without treating
    five overlapping outer training sets as independent stability replicates.
    """
    matrix = np.asarray([[name in subset for name in universe] for subset in sets], dtype=float)
    if len(sets) < 2 or any(not subset for subset in sets):
        raise ValueError("Stability requires at least two nonempty selected sets")
    if any(set(subset) - set(universe) for subset in sets):
        raise ValueError("Selected names must belong to the feature universe")
    q = matrix.mean()
    score = 1.0 if q in (0.0, 1.0) else 1 - float(matrix.var(axis=0, ddof=1).mean()) / (q * (1-q))
    jaccard = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(sets, 2)]
    return {"nogueira": score, "mean_pairwise_jaccard": float(np.mean(jaccard)),
            "n_fold_pairs": len(jaccard), "selection_counts": matrix.sum(axis=0).astype(int).tolist()}


def paired_patient_bootstrap(truth, predictions, groups, *, n_bootstrap=5000, seed=42):
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    truth = np.asarray(truth)
    if set(truth) != {0, 1}:
        raise ValueError("Both binary classes are required")
    unique, inverse = np.unique(groups, return_inverse=True)
    patient_labels = np.empty(len(unique), dtype=int)
    for i in range(len(unique)):
        labels = np.unique(truth[inverse == i])
        if len(labels) != 1:
            raise ValueError("Each patient must have a consistent diagnosis")
        patient_labels[i] = labels[0]
    strata = [np.flatnonzero(patient_labels == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = {method: np.empty(n_bootstrap) for method in predictions}
    codes = {}
    for method, pred in predictions.items():
        pred = np.asarray(pred)
        if pred.shape != truth.shape or not np.isin(pred, [0, 1]).all():
            raise ValueError("Prediction shape/labels do not match the target")
        codes[method] = 2 * truth + pred
    for b in range(n_bootstrap):
        drawn = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        weights = np.bincount(drawn, minlength=len(unique))[inverse]
        for method in predictions:
            samples[method][b] = metrics_from_counts(np.bincount(codes[method], weights=weights, minlength=4))
    intervals = {m: np.quantile(values, [.025, .975]).tolist() for m, values in samples.items()}
    differences = {}
    for first, second in combinations(predictions, 2):
        differences[f"{second}_minus_{first}"] = np.quantile(samples[second]-samples[first], [.025, .975]).tolist()
    return intervals, differences


def summarize(dataset, submissions, *, n_bootstrap=5000):
    table = dataset.table
    base = table[["image_id", "patient_id", "fold5_kfold", "diagnosis"]].copy()
    predictions, outputs = {}, {}
    for method in METHODS:
        location = Path(submissions) / method
        info = json.loads((location / "run_info.json").read_text())
        if (info.get("method") != method or info.get("dataset_sha256") != dataset.sha256
                or info.get("protocol_id") != read_contract()["protocol_id"] or info.get("status") != "COMPLETED"):
            raise ValueError(f"{method}: incompatible or incomplete experiment")
        pred = pd.read_csv(location / "predictions.csv")
        required = ["image_id", "patient_id", "outer_fold", "y_true", "y_pred"]
        if list(pred) != required or len(pred) != len(base) or pred.isna().any().any():
            raise ValueError(f"{method}: prediction schema, length or missing values")
        if not pred.y_pred.isin([0, 1]).all() or pred.image_id.duplicated().any():
            raise ValueError(f"{method}: invalid predictions")
        joined = base.merge(pred, on="image_id", validate="one_to_one", how="outer", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError(f"{method}: image identifiers differ")
        if not (joined.patient_id_x.eq(joined.patient_id_y).all() and joined.diagnosis.eq(joined.y_true).all()
                and joined.fold5_kfold.eq(joined.outer_fold).all()):
            raise ValueError(f"{method}: patient, truth or fold alignment differs")
        # Reindex explicitly: merge ordering must not determine paired inference.
        pred = pred.set_index("image_id").loc[base.image_id].reset_index()
        predictions[method] = pred.y_pred.to_numpy(dtype=int)
        selected = pd.read_csv(location / "selected_features.csv")
        if list(selected) != ["outer_fold", "feature_name"] or selected.isna().any().any() or selected.duplicated().any():
            raise ValueError(f"{method}: invalid selected-feature records")
        if set(selected.outer_fold) != {1, 2, 3, 4, 5}:
            raise ValueError(f"{method}: selected features must cover all five folds")
        sets = [set(selected.loc[selected.outer_fold == f, "feature_name"]) for f in range(1, 6)]
        if method == "baseline" and any(s != set(dataset.feature_names) for s in sets):
            raise ValueError("Baseline must retain every predictor")
        stability = selection_stability(sets, dataset.feature_names)
        timing = pd.read_csv(location / "timings.csv")
        if len(timing) != 5 or set(timing.outer_fold) != {1, 2, 3, 4, 5} or not np.isfinite(timing.total_seconds).all() or (timing.total_seconds <= 0).any():
            raise ValueError(f"{method}: invalid fold timings")
        folds = []
        for f, subset in enumerate(sets, 1):
            mask = pred.outer_fold.to_numpy() == f
            folds.append({"fold": f, "features": len(subset),
                          **classification_metrics(pred.y_true.to_numpy()[mask], predictions[method][mask])})
        metrics = {key: {"mean": float(np.mean([v[key] for v in folds])),
                         "sd": float(np.std([v[key] for v in folds], ddof=1))}
                   for key in ("accuracy", "f1_macro", "balanced_accuracy", "malignant_sensitivity", "benign_specificity")}
        outputs[method] = {"folds": folds, "metrics": metrics,
                           "pooled": classification_metrics(pred.y_true, predictions[method]),
                           "mean_features": float(np.mean([len(s) for s in sets])),
                           "feature_reduction_pct": float(100*(1-np.mean([len(s) for s in sets])/len(dataset.feature_names))),
                           "stability": stability, "total_wall_seconds": float(timing.total_seconds.sum()),
                           "versions": info["versions"], "platform": info["platform"]}
    intervals, differences = paired_patient_bootstrap(base.diagnosis.to_numpy(), predictions, base.patient_id.to_numpy(), n_bootstrap=n_bootstrap)
    for method in METHODS:
        outputs[method]["pooled_macro_f1_patient_bootstrap_95ci"] = intervals[method]
    effects = {}
    for first, second in combinations(METHODS, 2):
        key = f"{second}_minus_{first}"
        effects[key] = {"pooled_macro_f1_difference": outputs[second]["pooled"]["f1_macro"]-outputs[first]["pooled"]["f1_macro"],
                        "patient_bootstrap_95ci": differences[key]}
    return {"schema_version": 1, "protocol_id": read_contract()["protocol_id"], "dataset_sha256": dataset.sha256,
            "n_images": len(base), "n_patients": int(base.patient_id.nunique()), "n_predictors": len(dataset.feature_names),
            "bootstrap": {"replicates": n_bootstrap, "seed": 42, "unit": "patient", "stratified_by": "diagnosis",
                          "scope": "conditional on fixed out-of-fold predictions; not full-pipeline retraining uncertainty"},
            "stability_scope": "descriptive across five overlapping outer training sets; named features include redundant measurements",
            "methods": outputs, "paired_effects": effects}
