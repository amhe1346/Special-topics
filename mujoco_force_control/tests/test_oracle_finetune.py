import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_oracle_finetune.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_training_requires_checkpoint_terms_review():
    result = run_cli("--train", "--checkpoint-base-dir", "/tmp/unused", "--exp-name", "test")
    assert result.returncode != 0
    assert "training requires --terms-reviewed" in result.stderr


def test_training_requires_explicit_checkpoint_destination():
    result = run_cli("--train", "--terms-reviewed")
    assert result.returncode != 0
    assert "training requires --checkpoint-base-dir and --exp-name" in result.stderr


def test_invalid_experiment_name_is_rejected():
    result = run_cli("--exp-name", "../other")
    assert result.returncode != 0
    assert "--exp-name must be a single directory name" in result.stderr


def test_invalid_diagnostic_interval_is_rejected():
    result = run_cli("--diagnostic-interval-s", "-1")
    assert result.returncode != 0
    assert "--diagnostic-interval-s must be non-negative" in result.stderr


def test_incomplete_dataset_is_rejected_before_loading_dependencies(tmp_path):
    (tmp_path / "conversion_manifest.json").write_text('{"status":"in_progress"}')
    result = run_cli("--dataset", str(tmp_path))
    assert result.returncode != 0
    assert "expected a completed oracle LeRobot conversion" in result.stderr
