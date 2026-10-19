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
