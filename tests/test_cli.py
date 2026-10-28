import json
from types import SimpleNamespace

import pandas as pd

import busbra.cli as cli
from busbra.cli import main


def test_cli_rejects_missing_data_without_creating_results(tmp_path, capsys):
    output = tmp_path / "results"
    assert main(["baseline", "--data", str(tmp_path / "missing.csv"), "--output", str(output)]) == 2
    assert "Private dataset missing" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_changed_input(tmp_path, capsys):
    changed = tmp_path / "changed.csv"
    changed.write_text("changed input")
    assert main(["validate", "--data", str(changed)]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_existing_results_are_preserved_without_training(tmp_path, monkeypatch, capsys):
    output = tmp_path / "previous-run"
    output.mkdir()
    sentinel = output / "do-not-overwrite.txt"
    sentinel.write_text("preserve me")
    monkeypatch.setattr(cli, "load_dataset", lambda path: object())

    def forbidden_training(*args, **kwargs):
        raise AssertionError("Should reject existing output before starting expensive training")

    monkeypatch.setattr(cli, "evaluate_baseline", forbidden_training)
    assert main(["baseline", "--output", str(output)]) == 2
    assert sentinel.read_text() == "preserve me"
    assert "already exists" in capsys.readouterr().err


def test_cli_exports_official_submission(tmp_path, synthetic_table, monkeypatch):
    from busbra.data import ValidatedDataset
    from busbra.evaluation import evaluate_method
    output = tmp_path / "new-run"
    names = tuple(c for c in synthetic_table if c.startswith("original_"))
    dataset = ValidatedDataset(synthetic_table, names, "synthetic-test-only")
    monkeypatch.setattr(cli, "load_dataset", lambda path: dataset)
    assert main(["fuzzy_rough", "--output", str(output), "--jobs", "1"]) == 0
    assert len(pd.read_csv(output / "predictions.csv")) == len(synthetic_table)
    assert list(pd.read_csv(output / "best_params.csv")) == [
        "outer_fold", "kernel", "C", "gamma", "class_weight", "inner_macro_f1", "inner_accuracy"]
    assert list(pd.read_csv(output / "timings.csv")) == [
        "outer_fold", "selector_seconds", "svm_tuning_seconds", "refit_seconds", "predict_seconds", "total_seconds"]
    record = json.loads((output / "run_info.json").read_text())
    assert record["status"] == "COMPLETED"
    assert record["method"] == "fuzzy_rough"
    assert (output / "code/busbra/selectors.py").is_file()


def test_run_all_records_completed_and_does_not_overwrite(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    calls = []
    monkeypatch.setattr(cli, 'load_dataset', lambda path: SimpleNamespace(sha256='synthetic'))
    monkeypatch.setattr(cli, 'evaluate_method', lambda dataset, method, **kw: calls.append(method) or method)
    monkeypatch.setattr(cli, 'export_submission', lambda result, path: path.mkdir(parents=True))
    assert main(['run-all','--output',str(output)]) == 0
    assert calls == ['baseline','fuzzy_rough','gbfrs']
    record = json.loads((output/'experiment.json').read_text())
    assert record['status'] == 'COMPLETED'
    assert record['completed_methods'] == calls
    assert main(['run-all','--output',str(output)]) == 2
    assert len(calls) == 3


def test_run_all_preserves_failure_manifest(tmp_path, monkeypatch):
    output = tmp_path / 'failed'
    monkeypatch.setattr(cli,'load_dataset', lambda path: SimpleNamespace(sha256='synthetic'))
    def fail(*args, **kwargs):
        raise ValueError('synthetic failure')
    monkeypatch.setattr(cli,'evaluate_method',fail)
    assert main(['run-all','--output',str(output)]) == 2
    assert json.loads((output/'experiment.json').read_text())['status'] == 'FAILED'
