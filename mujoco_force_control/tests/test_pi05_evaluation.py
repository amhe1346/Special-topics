import json

import numpy as np
import pytest

import evaluate_pi05
from collect_scripted_teacher import _set_fixture_tilt
from envs.peg_in_hole import PegInHoleEnv
from integrations.rollout import RolloutConfig


class Renderer:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


class Policy:
    metadata = {"checkpoint": "mock"}

    def infer(self, observation):
        actions = np.zeros((15, 8))
        actions[:, 7] = 1
        return {"actions": actions}


def source_manifest(tmp_path):
    source = {
        "schema": "stage-c-oracle-v2",
        "status": "complete",
        "splits": {"train": [0], "validation": [100, 101], "test": [200]},
        "randomization": {
            "lateral_offset_range_m": [-0.00005, 0.00005],
            "fixture_tilt_xy_range_deg": [-0.03, 0.03],
        },
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    return path


def test_evaluation_matches_oracle_randomization_and_records_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_pi05.mujoco, "Renderer", Renderer)
    config = RolloutConfig(timeout_s=0.1, no_contact=False, open_loop_horizon=1)
    report = evaluate_pi05.evaluate(Policy(), source_manifest(tmp_path), tmp_path / "run", config=config)
    assert report["status"] == "complete"
    assert report["seeds"] == [100, 101]
    assert report["summary"]["trials"] == 2
    assert report["summary"]["successes"] == 0
    assert report["summary"]["outcomes"] == {"timeout": 2}
    assert report["warmup"]["inference_wall_s"] >= 0
    for trial in report["trials"]:
        env = PegInHoleEnv()
        env.reset(seed=trial["seed"], lateral_range=0.00005)
        _set_fixture_tilt(env, trial["seed"], 0.03)
        np.testing.assert_array_equal(trial["fixture_position"], env.data.mocap_pos[env.fixture_mocap_id])
        np.testing.assert_array_equal(trial["fixture_quaternion"], env.data.mocap_quat[env.fixture_mocap_id])
        assert trial["result"]["inference_timing"]["calls"] > 0
        force = trial["result"]["force_summary"]
        assert force["sample_count"] == trial["result"]["steps"]
        assert 0 <= force["mean_n"] <= force["peak_n"]
        assert 0 <= force["time_above_reference_s"] <= trial["result"]["sim_time_s"] + 1e-9
    saved = json.loads((tmp_path / "run/report.json").read_text())
    assert saved == report


def test_repeated_evaluation_preserves_fixture_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_pi05.mujoco, "Renderer", Renderer)
    config = RolloutConfig(timeout_s=0.1, no_contact=False, open_loop_horizon=1)
    report = evaluate_pi05.evaluate(
        Policy(), source_manifest(tmp_path), tmp_path / "run", config=config, repeats=2,
    )
    assert [(t["repeat"], t["seed"]) for t in report["trials"]] == [
        (0, 100), (0, 101), (1, 100), (1, 101),
    ]
    assert report["trials"][0]["fixture_position"] == report["trials"][2]["fixture_position"]
    assert report["trials"][0]["fixture_quaternion"] == report["trials"][2]["fixture_quaternion"]
    assert report["summary"]["per_seed"][0]["outcomes"] == ["timeout", "timeout"]
    assert len(report["summary"]["per_seed"][0]["peak_forces_n"]) == 2


def test_evaluation_records_errors_and_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_pi05.mujoco, "Renderer", Renderer)

    class Broken(Policy):
        def infer(self, observation):
            raise RuntimeError("model unavailable")

    source = source_manifest(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="model unavailable"):
        evaluate_pi05.evaluate(Broken(), source, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "error"
    assert report["error"]["message"] == "model unavailable"
    with pytest.raises(FileExistsError):
        evaluate_pi05.evaluate(Policy(), source, output)


def test_evaluation_rejects_overlapping_seeds(tmp_path):
    source = source_manifest(tmp_path)
    manifest = json.loads(source.read_text())
    manifest["splits"]["test"] = [100]
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_pi05.evaluate(Policy(), source, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_finetuned_horizon_adapter_checks_and_truncates_actions():
    pytest.importorskip("openpi")
    from run_openpi_local_server import HorizonAdapter

    class FineTuned:
        def __init__(self, actions):
            self.actions = actions

        def infer(self, observation):
            return {"actions": self.actions, "other": 1}

    actions = np.arange(16 * 8).reshape(16, 8)
    result = HorizonAdapter(FineTuned(actions)).infer({})
    np.testing.assert_array_equal(result["actions"], actions[:15])
    assert result["other"] == 1
    with pytest.raises(ValueError, match=r"\(16, 8\)"):
        HorizonAdapter(FineTuned(np.zeros((15, 8)))).infer({})
    invalid = actions.astype(float)
    invalid[-1, -1] = np.nan
    with pytest.raises(ValueError, match=r"\(16, 8\)"):
        HorizonAdapter(FineTuned(invalid)).infer({})
