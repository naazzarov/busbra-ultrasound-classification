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


def test_cli_writes_internal_artifacts_to_new_directory(tmp_path, monkeypatch):
    output = tmp_path / "new-run"
    result = SimpleNamespace(
        predictions=pd.DataFrame({"image_id": ["synthetic"], "y_pred": [0]}),
        selected_features=pd.DataFrame({"outer_fold": [1], "feature_name": ["original_0"]}),
        metadata={"method": "baseline", "fixture": "synthetic-software-test"},
        folds=[{"outer_fold": 1}],
    )
    monkeypatch.setattr(cli, "load_dataset", lambda path: object())
    monkeypatch.setattr(cli, "evaluate_baseline", lambda dataset, n_jobs: result)
    assert main(["baseline", "--output", str(output)]) == 0
    assert set(path.name for path in output.iterdir()) == {"predictions.csv", "selected_features.csv", "internal_run.json"}
    record = json.loads((output / "internal_run.json").read_text())
    assert record["folds"] == [{"outer_fold": 1}]
    assert record["fixture"] == "synthetic-software-test"
