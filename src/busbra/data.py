"""Verify the frozen CSV before exposing predictors to an experiment."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import read_contract

METADATA = {"image_id", "patient_id", "source_case", "diagnosis", "pathology", "fold5_kfold", "fold10_kfold"}


@dataclass(frozen=True)
class ValidatedDataset:
    table: pd.DataFrame
    feature_names: tuple[str, ...]
    sha256: str


def validate_table(table: pd.DataFrame, *, n_rows: int, n_patients: int, n_predictors: int) -> tuple[str, ...]:
    """Structural validation; production loading separately requires the exact hash."""
    if not table.columns.is_unique:
        raise ValueError("Duplicate column names are forbidden")
    names = tuple(name for name in table if name.startswith(("original_", "wavelet-")))
    if len(table) != n_rows or len(names) != n_predictors:
        raise ValueError(f"Expected {n_rows} rows and {n_predictors} radiomic predictors")
    if set(table) != METADATA | set(names):
        raise ValueError("Expected exactly the seven metadata columns plus radiomic predictors")
    if table[["image_id", "patient_id", "diagnosis", "fold5_kfold"]].isna().any().any():
        raise ValueError("Missing identifiers, diagnosis or outer-fold assignments are forbidden")
    if table.image_id.duplicated().any():
        raise ValueError("Image identifiers must be unique")
    if table.patient_id.nunique() != n_patients:
        raise ValueError(f"Expected {n_patients} patient groups")
    if not all(pd.api.types.is_numeric_dtype(table[name]) for name in names):
        raise ValueError("Every radiomic predictor must be numeric")
    if not np.isfinite(table[list(names)].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite radiomic predictors are forbidden")
    if set(table.diagnosis) != {0, 1}:
        raise ValueError("Diagnosis must contain both labels 0 and 1")
    if table.groupby("patient_id").diagnosis.nunique().max() != 1:
        raise ValueError("Patient groups must have a consistent diagnosis")
    contract = read_contract()
    if set(table.fold5_kfold) != set(contract["outer_folds"]):
        raise ValueError("Outer folds must be exactly 1 through 5")
    if table.groupby("patient_id").fold5_kfold.nunique().max() != 1:
        raise ValueError("Patient leakage: a patient appears in multiple outer folds")
    for fold in contract["outer_folds"]:
        if set(table.loc[table.fold5_kfold == fold, "diagnosis"]) != {0, 1}:
            raise ValueError(f"Outer fold {fold} must contain both classes")
    return names


def load_dataset(path: str | Path) -> ValidatedDataset:
    path = Path(path)
    contract = read_contract()
    if not path.is_file():
        raise FileNotFoundError(f"Private dataset missing: {path}. Supply the original team CSV.")
    # Hash and parse the same bytes: never train on a file changed after verification.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != contract["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 mismatch; use the unchanged original team CSV")
    from io import BytesIO

    table = pd.read_csv(BytesIO(raw), dtype={"image_id": str, "patient_id": str})
    names = validate_table(
        table, n_rows=contract["n_rows"], n_patients=contract["n_patients"],
        n_predictors=contract["n_predictors"],
    )
    return ValidatedDataset(table, names, digest)
