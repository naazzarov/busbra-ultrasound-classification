from pathlib import Path
import subprocess


def test_private_artifacts_are_ignored_at_arbitrary_depth():
    root = Path(__file__).parents[1]
    files = ["inputs/BUSBRA_features_ready.csv", "submissions/baseline/predictions.csv",
             "private_outputs/anything.md", "private_handoff/GBFRS.py",
             "accidental/export/selected_features.csv", "elsewhere/run_info.json",
             "elsewhere/internal_run.json", "BUS-BRA_Team_Brief.pdf"]
    for name in files:
        result = subprocess.run(["git", "check-ignore", "-q", name], cwd=root)
        assert result.returncode == 0, name


def test_publication_check_detects_identifiers_embedded_in_json(tmp_path, monkeypatch, capsys):
    import importlib.util
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('privacy_checker',root/'tools/check_public_files.py')
    checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    # Construct a synthetic identifier at runtime; no real subject data is a fixture.
    (tmp_path/'notes.json').write_text('{"patient_id":"'+'case_'+format(123,'04d')+'"}')
    subprocess.run(['git','add','-f','notes.json'],cwd=tmp_path,check=True)
    monkeypatch.chdir(tmp_path)
    assert checker.check()==1
    assert 'blocked' in capsys.readouterr().err
