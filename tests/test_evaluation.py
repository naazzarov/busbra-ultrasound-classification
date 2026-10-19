import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from busbra.data import ValidatedDataset, validate_table
import busbra.evaluation as evaluation
from busbra.protocol import read_contract, svm_grid


def test_packaged_contract_matches_team_contract():
    source = Path(__file__).parents[1] / "evaluation/evaluation_contract.json"
    assert read_contract() == json.loads(source.read_text())
    assert len(list(ParameterGrid(svm_grid()))) == 32
    assert read_contract()["n_predictors"] == 288


def test_clinical_metrics_have_correct_positive_class():
    metrics = evaluation.classification_metrics([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])
    assert metrics["malignant_sensitivity"] == 0.5
    assert metrics["benign_specificity"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == 0.6
    with pytest.raises(ValueError, match="Both classes"):
        evaluation.classification_metrics([0, 0], [0, 0])


def test_full_baseline_keeps_every_scaler_inside_training(synthetic_table, monkeypatch):
    fit_groups = []

    class AuditedScaler(StandardScaler):
        def fit(self, X, y=None, **kwargs):
            fit_groups.append(set(X[:, 0].astype(int)))
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(evaluation, "StandardScaler", AuditedScaler)
    names = validate_table(synthetic_table, n_rows=120, n_patients=60, n_predictors=288)
    dataset = ValidatedDataset(synthetic_table, names, "synthetic-software-test-only")
    result = evaluation.evaluate_baseline(dataset, n_jobs=1)
    assert len(result.predictions) == 120
    assert result.predictions.image_id.is_unique
    assert result.predictions.y_pred.isin([0, 1]).all()
    assert result.predictions.y_true.equals(synthetic_table.diagnosis)
    assert len(result.selected_features) == 5 * 288
    assert len(result.folds) == 5
    assert [fold["inner_seed"] for fold in result.folds] == [43, 44, 45, 46, 47]
    # 32 candidates x 3 CV fits plus one refit for each outer fold.
    assert len(fit_groups) == 5 * 97
    for index, fold in enumerate(result.folds):
        train_groups = set(synthetic_table.loc[synthetic_table.fold5_kfold != index + 1, "patient_id"])
        observed = fit_groups[index * 97:(index + 1) * 97]
        assert all(groups <= train_groups for groups in observed)
        assert all(len(groups) < len(train_groups) for groups in observed[:-1])
        assert observed[-1] == train_groups
        assert fold["n_candidates"] == 32
        assert fold["search_and_refit_seconds"] > 0
        assert fold["prediction_seconds"] > 0
        assert np.isfinite(list(fold["metrics"].values())).all()
        assert fold["selected_feature_count"] == 288
    assert "sample_sd" in result.metadata["summary"]["f1_macro"]
    json.dumps(result.metadata, allow_nan=False)
    json.dumps(result.folds, allow_nan=False)


def test_evaluation_rechecks_outer_leakage(synthetic_table):
    synthetic_table.loc[1, "fold5_kfold"] = 2
    names = tuple(f"original_{i}" for i in range(288))
    with pytest.raises(ValueError, match="Patient leakage"):
        evaluation.evaluate_baseline(ValidatedDataset(synthetic_table, names, "synthetic"))
