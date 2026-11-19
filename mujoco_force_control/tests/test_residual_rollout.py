from dataclasses import replace

import numpy as np
import pytest

from envs.peg_episode import PegEpisode
from envs.peg_in_hole import PegInHoleEnv
from integrations.residual_adapter import ForceResidualObservation
from integrations.residual_experiments import Variant, get_config
from integrations.rollout import ResidualRollout, RolloutConfig, run_rollout
from sensors.residual_wrench import ResidualWrenchInput, WrenchInputError


class Renderer:
    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


class Nominal:
    def infer(self, observation):
        actions = np.zeros((15, 8))
        actions[:, 0] = 0.01
        actions[:, 7] = 1
        return {"actions": actions}


class Residual:
    def __init__(self, value=0):
        self.value = value
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        return np.full(7, self.value)


def run(residual=None):
    env = PegInHoleEnv()
    env.reset(seed=0, lateral_range=0.00005)
    return run_rollout(
        env, Renderer(), Nominal(), RolloutConfig(timeout_s=0.3, no_contact=False),
        residual=residual,
    )


@pytest.mark.parametrize("variant", list(Variant))
def test_zero_residual_matches_nominal_simulation(variant):
    policy = Residual()
    baseline = run()
    result = run(ResidualRollout(get_config(variant), policy, 0))
    assert result["reason"] == baseline["reason"] == "timeout"
    assert result["steps"] == baseline["steps"]
    assert result["depth_m"] == pytest.approx(baseline["depth_m"], abs=1e-12)
    assert result["peak_force_n"] == pytest.approx(baseline["peak_force_n"], abs=1e-12)
    original = [a for d in baseline["decisions"] for a in d["executed_actions"]]
    changed = [a for d in result["decisions"] for a in d["executed_actions"]]
    assert len(changed) == len(original) == len(policy.observations)
    for before, after in zip(original, changed, strict=True):
        np.testing.assert_allclose(before["joint_target_rad"], after["joint_target_rad"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(after["residual"]["bounded_delta_rad"], np.zeros(7))
    assert all(isinstance(obs, ForceResidualObservation) == get_config(variant).force_observation
               for obs in policy.observations)


def test_nonzero_mock_correction_clips_and_is_recorded_every_tick():
    policy = Residual(0.1)
    result = run(ResidualRollout(get_config(Variant.OBSERVATION_ONLY), policy, 0))
    actions = [a for decision in result["decisions"] for a in decision["executed_actions"]]
    assert result["reason"] == "timeout"
    assert len(actions) == len(policy.observations) > 1
    for action in actions:
        record = action["residual"]
        np.testing.assert_allclose(record["bounded_delta_rad"], 0.001)
        assert all(record["residual_clipped"])
        assert action["simulation_time_s"] - record["wrench_time_s"] <= 0.1
        assert np.max(np.abs(action["joint_delta_rad"])) <= 0.1 / 15 + 1e-12


@pytest.mark.parametrize("variant", [Variant.BASELINE, Variant.OBSERVATION_ONLY])
def test_sensor_fault_stops_and_records_reason(monkeypatch, variant):
    original = ResidualWrenchInput.update

    def fail_after_settle(self, time_s):
        if time_s >= 0.5:
            raise WrenchInputError("simulated sensor failure")
        original(self, time_s)

    monkeypatch.setattr(ResidualWrenchInput, "update", fail_after_settle)
    result = run(ResidualRollout(get_config(variant), Residual(), 0))
    assert result["reason"] == "wrench_input_error"
    assert result["residual_experiment"]["sensor_error"] == "simulated sensor failure"
    assert result["steps"] == 0
    assert result["action_steps"] == 0


@pytest.mark.parametrize("variant", list(Variant))
def test_stale_wrench_stops_before_residual_policy(monkeypatch, variant):
    def stale_sample(self, time_s):
        raise WrenchInputError("wrist wrench is stale")

    monkeypatch.setattr(ResidualWrenchInput, "sample", stale_sample)
    policy = Residual()
    result = run(ResidualRollout(get_config(variant), policy, 0))
    assert result["reason"] == "wrench_input_error"
    assert result["residual_experiment"]["sensor_error"] == "wrist wrench is stale"
    assert result["steps"] == result["action_steps"] == 0
    assert not policy.observations


def test_invalid_residual_policy_error_propagates():
    with pytest.raises(ValueError, match="residual output"):
        run(ResidualRollout(get_config(Variant.BASELINE), Residual(np.nan), 0))


def test_residual_rejects_mismatched_nominal_limits():
    env = PegInHoleEnv()
    with pytest.raises(ValueError, match="15 Hz"):
        run_rollout(
            env, Renderer(), Nominal(), RolloutConfig(frequency_hz=10),
            residual=ResidualRollout(get_config(Variant.BASELINE), Residual(), 0),
        )


@pytest.mark.parametrize("variant", list(Variant))
def test_reward_accounts_for_every_physics_step_and_terminal_once(variant):
    result = run(ResidualRollout(get_config(variant), Residual(), 0))
    experiment = result["residual_experiment"]
    intervals = experiment["reward_intervals"]
    assert experiment["reward_status"] == "complete"
    assert experiment["reward_implemented"] is True
    assert experiment["training_implemented"] is False
    assert len(intervals) == result["action_steps"]
    assert sum(interval["physics_steps"] for interval in intervals) == result["steps"]
    assert all(interval["breakdown"]["terminal"] == 0 for interval in intervals[:-1])
    assert intervals[-1]["termination"] == "timeout"
    assert intervals[-1]["breakdown"]["terminal"] == -5
    assert intervals[-1]["physics_steps"] < 33
    assert experiment["undiscounted_reward_total"] == pytest.approx(
        sum(interval["breakdown"]["total"] for interval in intervals)
    )
    executions = [item for decision in result["decisions"] for item in decision["executed_actions"]]
    assert [item["reward"] for item in executions] == [item["breakdown"] for item in intervals]


@pytest.mark.parametrize("variant", list(Variant))
def test_force_reward_factor_uses_terminal_physics_sample(monkeypatch, variant):
    original = PegEpisode.step

    def injected_force(self):
        reason = original(self)
        if self.samples:
            self.samples[-1] = replace(self.samples[-1], force_n=6.0)
            self.peak_force_n = 6.0
            self.reason = "force_limit"
            return self.reason
        return reason

    monkeypatch.setattr(PegEpisode, "step", injected_force)
    result = run(ResidualRollout(get_config(variant), Residual(), 0))
    assert result["reason"] == "force_limit"
    intervals = result["residual_experiment"]["reward_intervals"]
    assert len(intervals) == 1 and intervals[0]["physics_steps"] == 1
    breakdown = intervals[0]["breakdown"]
    assert breakdown["terminal"] == -5
    assert breakdown["force_excess_n_s"] == pytest.approx((6 - 1) * 0.002)
    assert breakdown["force_penalty"] == pytest.approx(
        -0.01 if get_config(variant).force_aware_reward else 0
    )


def test_invalid_mid_interval_is_not_scored(monkeypatch):
    original = ResidualWrenchInput.update

    def fail_during_action(self, time_s):
        if time_s >= 0.51:
            raise WrenchInputError("simulated failure")
        original(self, time_s)

    monkeypatch.setattr(ResidualWrenchInput, "update", fail_during_action)
    result = run(ResidualRollout(get_config(Variant.COMBINED), Residual(), 0))
    reward = result["residual_experiment"]
    assert result["reason"] == "wrench_input_error" and result["steps"] > 0
    assert reward["reward_status"] == "invalid"
    assert reward["reward_intervals"] == []
    assert reward["undiscounted_reward_total"] is None


def test_nominal_release_is_separate_terminal_event():
    class Release(Nominal):
        def infer(self, observation):
            return {"actions": np.zeros((15, 8))}

    env = PegInHoleEnv()
    env.reset(seed=0)
    result = run_rollout(
        env, Renderer(), Release(), RolloutConfig(timeout_s=0.3, no_contact=False),
        residual=ResidualRollout(get_config(Variant.BASELINE), Residual(), 0),
    )
    assert result["reason"] == "release_requested"
    assert result["action_steps"] == 0
    event = result["residual_experiment"]["reward_intervals"][0]
    assert event["kind"] == "nominal_release_event"
    assert event["physics_steps"] == 0
    assert event["breakdown"]["terminal"] == -5


def test_success_terminal_is_scored_once(monkeypatch):
    original = PegEpisode.step

    def simulated_success(self):
        reason = original(self)
        if self.samples:
            self.reason = "success"
            return self.reason
        return reason

    monkeypatch.setattr(PegEpisode, "step", simulated_success)
    result = run(ResidualRollout(get_config(Variant.BASELINE), Residual(), 0))
    intervals = result["residual_experiment"]["reward_intervals"]
    assert result["reason"] == "success"
    assert len(intervals) == 1
    assert intervals[0]["physics_steps"] == 1
    assert intervals[0]["breakdown"]["terminal"] == 10


def test_invalid_after_valid_interval_has_no_trial_total(monkeypatch):
    original = ResidualWrenchInput.update

    def fail_after_one_interval(self, time_s):
        if time_s >= 0.58:
            raise WrenchInputError("simulated failure")
        original(self, time_s)

    monkeypatch.setattr(ResidualWrenchInput, "update", fail_after_one_interval)
    result = run(ResidualRollout(get_config(Variant.BASELINE), Residual(), 0))
    reward = result["residual_experiment"]
    assert result["reason"] == "wrench_input_error"
    assert reward["reward_status"] == "invalid"
    assert len(reward["reward_intervals"]) == 1
    assert reward["reward_intervals"][0]["termination"] is None
    assert reward["undiscounted_reward_total"] is None


def test_residual_reward_rejects_contact_stop_mode():
    env = PegInHoleEnv()
    with pytest.raises(ValueError, match="contact to be allowed"):
        run_rollout(
            env, Renderer(), Nominal(), RolloutConfig(),
            residual=ResidualRollout(get_config(Variant.BASELINE), Residual(), 0),
        )
