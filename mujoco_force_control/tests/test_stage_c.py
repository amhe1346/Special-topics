import json

import mujoco
import numpy as np
import pytest

from collect_stage_c import Recorder, collect
from controllers.pd_position_controller import PDPositionController
from envs.peg_in_hole import PegInHoleEnv
from integrations.rollout import RolloutConfig, run_rollout
from validate_stage_c_conversion import evaluate_trial, motion_action


class Renderer:
    def __init__(self, *args, **kwargs):
        pass

    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def close(self):
        pass


class Policy:
    def __init__(self, value=0.0, closed=True):
        self.value = value
        self.closed = closed

    def infer(self, observation):
        result = np.zeros((15, 8))
        result[:, 0] = self.value
        result[:, 7] = float(self.closed)
        return {"actions": result}


def test_rollout_simulation_only_bounded_speed_and_timeout():
    env = PegInHoleEnv()
    result = run_rollout(env, Renderer(), Policy(1.0), RolloutConfig(timeout_s=0.2))
    assert result["reason"] == "timeout"
    assert result["fixture_unchanged"]
    assert result["sim_time_s"] == pytest.approx(0.2)
    assert len(result["inference_s"]) == 3
    assert result["steps"] == 100


def test_rollout_stops_before_unexpected_release():
    result = run_rollout(PegInHoleEnv(), Renderer(), Policy(closed=False), RolloutConfig())
    assert result["reason"] == "release_requested"
    assert result["steps"] == 0


def test_stale_observation_aborts_before_action():
    env = PegInHoleEnv()

    class BadPolicy(Policy):
        def infer(self, observation):
            env.step()
            return super().infer(observation)

    result = run_rollout(env, Renderer(), BadPolicy(), RolloutConfig())
    assert result["reason"] == "stale_observation"
    assert result["steps"] == 0


def test_inference_exception_clears_arm_controls():
    env = PegInHoleEnv()

    class BrokenPolicy(Policy):
        def infer(self, observation):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        run_rollout(env, Renderer(), BrokenPolicy(), RolloutConfig())
    ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
           for i in range(1, 8)]
    assert all(idx >= 0 for idx in ids)
    np.testing.assert_array_equal(env.data.ctrl[ids], np.zeros(7))


def test_rollout_records_steps_and_policy_decisions():
    env = PegInHoleEnv()
    sampled = []
    observations = []
    result = run_rollout(env, Renderer(), Policy(1), RolloutConfig(timeout_s=0.1),
                         on_step=lambda current: sampled.append(float(current.data.time)),
                         on_observation=observations.append)
    assert len(sampled) == result["steps"]
    assert np.all(np.diff(sampled) > 0)
    decision = result["decisions"][0]
    assert len(decision["normalized_velocity"]) == 7
    assert len(decision["joint_target_rad"]) == 7
    assert [observation.time_s for observation in observations] == [
        decision["observation_time_s"] for decision in result["decisions"]]


def test_inference_timeout_is_recorded_and_stops_before_action():
    class TimedOutPolicy(Policy):
        def infer(self, observation):
            raise TimeoutError("no response")

    result = run_rollout(PegInHoleEnv(), Renderer(), TimedOutPolicy(), RolloutConfig())
    assert result["reason"] == "inference_timeout"
    assert result["steps"] == 0
    assert len(result["inference_s"]) == 1


def test_invalid_sim_time_is_json_serializable(monkeypatch):
    def invalid_step(episode):
        episode.env.data.time = float("nan")
        return "invalid_state"

    monkeypatch.setattr("integrations.rollout.PegEpisode.step", invalid_step)
    result = run_rollout(PegInHoleEnv(), Renderer(), Policy(), RolloutConfig())
    assert result["reason"] == "invalid_state"
    assert result["sim_time_s"] is None
    json.dumps(result, allow_nan=False)


def test_splits_reject_overlap_before_writing(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        collect(tmp_path / "data", {"train": [1], "validation": [1], "test": [2]})
    assert not (tmp_path / "data").exists()


def test_training_only_normalization_and_trial_schema(tmp_path, monkeypatch):
    def fake_trial(method, config, on_step):
        row = dict(time_s=1., exterior_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                   wrist_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                   joint_position_rad=np.full(7, config.seed, dtype=float))
        on_step.rows.append(row)
        on_step.steps.append({"time_s": 1.})
        return {"reason": "success"}

    monkeypatch.setattr("collect_stage_c.run_trial", fake_trial)
    path = tmp_path / "data"
    collect(path, {"train": [1, 3], "validation": [20], "test": [30]})
    norm = json.loads((path / "normalization.json").read_text())
    assert norm["mean"] == [2.0] * 7
    assert norm["std"] == [1.0] * 7
    assert norm["training_seeds"] == [1, 3]
    with np.load(path / "validation-20.npz", allow_pickle=False) as trial:
        assert trial["joint_position_rad"][0, 0] == 20
    with np.load(path / "validation-20-steps.npz", allow_pickle=False) as trial:
        assert trial["time_s"][0] == 1
    with pytest.raises(FileExistsError):
        collect(path, {"train": [1], "validation": [2], "test": [3]})


def test_failed_collection_marks_partial_manifest(tmp_path, monkeypatch):
    def broken_trial(*args, **kwargs):
        raise RuntimeError("collection failed")

    monkeypatch.setattr("collect_stage_c.run_trial", broken_trial)
    path = tmp_path / "data"
    with pytest.raises(RuntimeError, match="collection failed"):
        collect(path, {"train": [1], "validation": [2], "test": [3]})
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "in_progress"
    assert manifest["trials"] == []


def test_motion_action_uses_achieved_motion_and_actual_elapsed_time():
    start = np.zeros(7)
    action = motion_action(start, np.full(7, 0.02), 0.1)
    np.testing.assert_allclose(action, np.full(7, 0.02 / (0.1 * 15 * 0.2)))
    with pytest.raises(ValueError):
        motion_action(start, start, 0)
    with pytest.raises(ValueError):
        motion_action(start, np.full(7, np.nan), 0.1)


def test_recorder_step_timing_and_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("collect_stage_c.mujoco.Renderer", Renderer)
    env = PegInHoleEnv()
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    recorder = Recorder(period_s=1 / 15)
    for _ in range(70):
        controller.step(target)
        env.step()
        mujoco.mj_forward(env.model, env.data)
        recorder(env)
    recorder.close()
    samples_path, steps_path = tmp_path / "samples.npz", tmp_path / "steps.npz"
    recorder.save(samples_path, steps_path)
    with np.load(samples_path) as samples, np.load(steps_path) as steps:
        assert len(steps["time_s"]) == 70
        assert len(samples["time_s"]) >= 2
        indices = samples["step_index"]
        np.testing.assert_allclose(samples["time_s"], steps["time_s"][indices])
        np.testing.assert_allclose(steps["time_s"] - steps["control_start_time_s"],
                                   env.model.opt.timestep)
        assert steps["executed_ctrl"].shape == (70, env.model.nu)
        assert steps["integration_state"].shape[0] == 70
    result = evaluate_trial(samples_path, steps_path, RolloutConfig())
    assert result["intervals"] == len(recorder.rows) - 1
    assert np.isfinite(result["p95_joint_error_rad"])
    assert np.isfinite(result["p95_tip_error_m"])
    assert result["reference_replay_max_joint_error_rad"] < 1e-12
    assert result["reference_replay_max_tip_error_m"] < 1e-12
    refined = evaluate_trial(samples_path, steps_path, RolloutConfig(), refine_iterations=1)
    assert refined["intervals"] == result["intervals"]
