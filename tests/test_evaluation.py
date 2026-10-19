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
    # Three training-only transforms, reused within split, plus outer refit.
    assert len(fit_groups) == 5 * 4
    for index, fold in enumerate(result.folds):
        train_groups = set(synthetic_table.loc[synthetic_table.fold5_kfold != index + 1, "patient_id"])
        observed = fit_groups[index * 4:(index + 1) * 4]
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


def test_candidate_tie_breaking():
    rows = [dict(macro_f1=.8, accuracy=.9, n_features=3),
            dict(macro_f1=.8, accuracy=.9, n_features=2),
            dict(macro_f1=.8, accuracy=.9, n_features=2)]
    assert evaluation.choose_candidate(rows) == 1
    rows[0]["accuracy"] = .91
    assert evaluation.choose_candidate(rows) == 0
    rows[2]["macro_f1"] = .81
    assert evaluation.choose_candidate(rows) == 2


@pytest.mark.parametrize("method", ["fuzzy_rough", "gbfrs"])
def test_selectors_are_fitted_only_on_current_training_partition(synthetic_table, monkeypatch, method):
    observed = []
    original = evaluation.make_selector
    # A scaler that retains the synthetic group marker lets the selector audit
    # its actual inputs; real scaling is independently audited above.
    class MarkerScaler:
        def fit(self, X):
            return self
        def transform(self, X):
            return X
    class Spy:
        def fit_transform(self, X, y):
            observed.append(set(X[:,0].astype(int)))
            self.impl = original(method).fit(X, y)
            self.dependency_history_ = self.impl.dependency_history_
            if method == "gbfrs":
                self.ball_sizes_ = self.impl.ball_sizes_
            return self.impl.transform(X)
        def get_support(self, indices):
            return self.impl.get_support(indices=indices)
        def transform(self, X):
            return self.impl.transform(X)
    monkeypatch.setattr(evaluation, "StandardScaler", MarkerScaler)
    monkeypatch.setattr(evaluation, "make_selector", lambda name: Spy())
    names = tuple(c for c in synthetic_table if c.startswith("original_"))
    result = evaluation.evaluate_method(ValidatedDataset(synthetic_table, names, "synthetic"), method=method, n_jobs=1)
    assert len(observed) == 20
    for f in range(1,6):
        expected = set(synthetic_table.loc[synthetic_table.fold5_kfold != f, "patient_id"])
        for inner_groups in observed[(f-1)*4:(f-1)*4+3]:
            assert inner_groups < expected
        assert observed[(f-1)*4+3] == expected
    assert len(result.predictions) == len(synthetic_table)
