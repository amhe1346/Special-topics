import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

import inspect_demonstrations
from envs.peg_in_hole import PegInHoleEnv
from integrations.rollout import RolloutConfig
from probe_realtime_pi05 import probe


class Renderer:
    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


class DelayedPolicy:
    def infer(self, observation):
        time.sleep(0.08)
        actions = np.zeros((15, 8))
        actions[:, 7] = 1
        return {"actions": actions}


def test_timing_probe_advances_physics_during_inference():
    env = PegInHoleEnv()
    result = probe(
        env, Renderer(), DelayedPolicy(),
        RolloutConfig(timeout_s=0.4, no_contact=False, open_loop_horizon=2),
    )
    assert result["reason"] == "timeout"
    assert result["sim_s"] >= 0.4
    assert result["deliveries"]
    assert result["deliveries"][0]["observation_age_sim_s"] >= 0.06
    assert result["inquiries"][0]["inference_wall_s"] >= 0.08
    assert result["deliveries"][0]["action_index"] == 0
    assert result["peak_force_n"] >= 0


def test_demonstration_inspection_reads_only_train(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "schema": "stage-c-oracle-v2",
        "status": "complete",
        "splits": {"train": [0], "validation": [100], "test": [200]},
        "trials": [
            {"split": "train", "seed": 0, "file": "train-0.npz"},
            {"split": "test", "seed": 200, "file": "test-200.npz"},
        ],
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    actions = np.zeros((3, 8))
    actions[:, 7] = 1
    image = np.zeros((3, 224, 224, 3), dtype=np.uint8)

    def read_trial(path, trial):
        assert path.name == "train-0.npz"
        return {"action": actions, "exterior_rgb": image, "wrist_rgb": image,
                "time_s": np.arange(3, dtype=float)}

    monkeypatch.setattr(inspect_demonstrations, "_read_trial", read_trial)
    report = inspect_demonstrations.inspect(source, tmp_path / "output")
    assert report["frames"] == 3
    assert report["gripper_closed_fraction_of_labels"] == 1
    assert (tmp_path / "output/training_camera_examples.png").is_file()


@pytest.mark.parametrize(
    ("mode", "reviewed", "expected"),
    [(0o700, "no", "TERMS_REVIEWED must be yes"), (0o755, "yes", "must not grant group or other access")],
)
def test_alpine_job_rejects_unreviewed_or_nonprivate_workdir(tmp_path, mode, reviewed, expected):
    private = tmp_path / "work"
    private.mkdir(mode=mode)
    private.chmod(mode)
    script = Path(__file__).resolve().parents[1] / "alpine_pi05_smoke.slurm"
    environment = {
        **os.environ,
        "PRIVATE_WORK_DIR": str(private),
        "EXPERIMENT_NAME": "test",
        "TERMS_REVIEWED": reviewed,
    }
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
