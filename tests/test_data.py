import numpy as np
import pytest
import hashlib

import busbra.data as data_module
from busbra.data import load_dataset, validate_table


def validate(table):
    return validate_table(table, n_rows=120, n_patients=60, n_predictors=288)


def test_only_original_features_are_available(synthetic_table):
    features = validate(synthetic_table)
    assert len(features) == 288
    assert "diagnosis" not in features
    assert "patient_id" not in features
    assert "fold5_kfold" not in features


@pytest.mark.parametrize("problem", ["leakage", "duplicate_image", "extra_column", "missing_feature", "infinite", "missing", "text", "wrong_label", "wrong_fold"])
def test_invalid_data_is_rejected(synthetic_table, problem):
    table = synthetic_table.copy()
    if problem == "leakage":
        table.loc[1, "fold5_kfold"] = 2
    elif problem == "duplicate_image":
        table.loc[1, "image_id"] = table.loc[0, "image_id"]
    elif problem == "extra_column":
        table["engineered_feature"] = 0
    elif problem == "missing_feature":
        table = table.drop(columns="original_287")
    elif problem == "infinite":
        table.loc[0, "original_1"] = np.inf
    elif problem == "missing":
        table.loc[0, "patient_id"] = np.nan
    elif problem == "text":
        table["original_1"] = "not-a-number"
    elif problem == "wrong_label":
        table.loc[0, "diagnosis"] = 2
    elif problem == "wrong_fold":
        table.loc[0, "fold5_kfold"] = 6
    with pytest.raises(ValueError):
        validate(table)


def test_duplicate_column_is_rejected(synthetic_table):
    table = synthetic_table.copy()
    table.columns = ["original_0" if col == "original_1" else col for col in table]
    with pytest.raises(ValueError, match="Duplicate column"):
        validate(table)


def test_missing_private_dataset_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Private dataset missing"):
        load_dataset(tmp_path / "absent.csv")


def test_substitute_dataset_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "substitute.csv"
    path.write_bytes(b"this is not the verified BUS-BRA CSV")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_dataset(path)


def test_loader_checks_and_parses_the_same_csv_bytes(tmp_path, synthetic_table, monkeypatch):
    path = tmp_path / "synthetic-software-test.csv"
    synthetic_table["image_id"] = [f"{i:05d}" for i in range(120)]
    synthetic_table.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = data_module.read_contract()
    # Fixture fingerprint only; the production contract and CLI have no bypass.
    contract.update(dataset_sha256=digest, n_rows=120, n_patients=60)
    monkeypatch.setattr(data_module, "read_contract", lambda: contract)
    dataset = load_dataset(path)
    assert dataset.sha256 == digest
    assert dataset.table.image_id.iloc[0] == "00000"
    assert len(dataset.feature_names) == 288
