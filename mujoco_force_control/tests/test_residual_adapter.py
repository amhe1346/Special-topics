from dataclasses import fields

import numpy as np
import pytest

from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import ActionPreview, preview_actions
from integrations.residual_adapter import (
    ForceResidualObservation,
    ResidualAdapter,
    ResidualObservation,
)
from sensors.load_cell import EndEffectorLoadCell
from sensors.residual_wrench import ResidualWrenchInput, WrenchInputError, WrenchSample


class MockWrench:
    def __init__(self):
        self.calls = []
        self.value = np.arange(6.0)
        self.sample_time = None

    def sample(self, time_s):
        self.calls.append(time_s)
        return WrenchSample(time_s if self.sample_time is None else self.sample_time, self.value.copy())


class MockResidual:
    def __init__(self, value=None):
        self.value = np.zeros(7) if value is None else value
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        return self.value


def action(value=0.0, closed=True):
    row = np.full((15, 8), value)
    row[:, 7] = 1.0 if closed else 0.0
    return preview_actions({"actions": row})[0]


def adapter(policy=None, wrench=None, force=False):
    return ResidualAdapter(
        policy or MockResidual(), wrench or MockWrench(), force_observation=force,
        start_time_s=0.5, physics_timestep_s=0.002,
    )


def tick(instance, time_s=0.5, nominal=None, joints=None, bounds=None):
    return instance.step(
        time_s=time_s, joint_position_rad=np.zeros(7) if joints is None else joints,
        joint_velocity_rad_s=np.arange(7.0), gripper_closed_fraction=0.75,
        nominal=action() if nominal is None else nominal,
        joint_bounds_rad=np.tile([-1.0, 1.0], (7, 1)) if bounds is None else bounds,
    )


@pytest.mark.parametrize("nominal_value", [-1.0, -0.02, 0.0, 0.02, 1.0])
def test_zero_residual_matches_existing_nominal_target(nominal_value):
    joints = np.linspace(-0.9, 0.9, 7)
    bounds = np.tile([-1.0, 1.0], (7, 1))
    nominal = action(nominal_value)
    expected = np.clip(joints + np.clip(nominal.joint_delta_rad, -0.1 / 15, 0.1 / 15),
                       bounds[:, 0], bounds[:, 1])
    result = tick(adapter(), nominal=nominal, joints=joints, bounds=bounds)
    np.testing.assert_array_equal(result.target_rad, expected)
    np.testing.assert_array_equal(result.executed_delta_rad, expected - joints)


def test_correction_speed_and_joint_bounds_are_enforced_and_recorded():
    policy = MockResidual(np.array([1.0, -1.0, 0.0005, 0, 0, 0, 0]))
    joints = np.array([0.0, 0.0, 0.9998, 0, 0, 0, 0])
    result = tick(adapter(policy=policy), nominal=action(0.5), joints=joints)
    np.testing.assert_allclose(result.residual_delta_rad[:3], [0.001, -0.001, 0.0005])
    np.testing.assert_array_equal(result.residual_clipped[:3], [True, True, False])
    assert result.total_delta_clipped[0]
    assert result.joint_limit_clipped[2]
    assert result.target_rad[2] == 1
    assert np.all(np.abs(result.executed_delta_rad) <= 0.1 / 15 + 1e-12)
    assert result.target_rad[1] == pytest.approx(0.1 / 15 - 0.001)


def test_each_tick_reuses_nominal_chunk_with_new_observation():
    policy, wrench = MockResidual(), MockWrench()
    instance = adapter(policy, wrench, force=True)
    chunk = preview_actions({"actions": np.tile(np.r_[np.zeros(7), 1], (15, 1))})
    for i in range(8):
        time = 0.5 + round((i / 15) / 0.002) * 0.002
        wrench.value = np.full(6, i)
        tick(instance, time, chunk[i])
    assert len(policy.observations) == len(wrench.calls) == 8
    assert all(isinstance(obs, ForceResidualObservation) for obs in policy.observations)
    for i, obs in enumerate(policy.observations):
        np.testing.assert_array_equal(obs.wrench_local, np.full(6, i))
        assert obs.wrench_time_s == wrench.calls[i]
    with pytest.raises(ValueError, match="early, late, or repeated"):
        tick(instance, wrench.calls[-1])


def test_force_off_policy_never_receives_wrench_but_health_gate_remains():
    policy, wrench = MockResidual(), MockWrench()
    instance = adapter(policy, wrench, force=False)
    tick(instance)
    obs = policy.observations[0]
    assert type(obs) is ResidualObservation
    assert {field.name for field in fields(obs)} == {
        "time_s", "joint_position_rad", "joint_velocity_rad_s", "gripper_closed_fraction",
        "nominal_delta_rad", "nominal_gripper_closed",
    }
    assert not hasattr(obs, "wrench_local") and not hasattr(obs, "wrench_time_s")
    assert wrench.calls == [0.5]
    wrench.sample_time = 0.0
    with pytest.raises(WrenchInputError, match="stale"):
        tick(instance, 0.566)
    assert len(policy.observations) == 1


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("kind", ["stale", "future", "nonfinite", "shape"])
def test_bad_sensor_stops_both_variants_before_policy(force, kind):
    policy, wrench = MockResidual(), MockWrench()
    if kind == "stale":
        wrench.sample_time = 0.3
    elif kind == "future":
        wrench.sample_time = 0.6
    elif kind == "nonfinite":
        wrench.value[0] = np.nan
    else:
        wrench.value = np.zeros(5)
    with pytest.raises(WrenchInputError):
        tick(adapter(policy, wrench, force))
    assert not policy.observations


@pytest.mark.parametrize("output", [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_bad_residual_output_is_an_error(output):
    with pytest.raises(ValueError, match="residual output"):
        tick(adapter(MockResidual(output)))


def test_invalid_inputs_and_nominal_actions_are_rejected():
    with pytest.raises(ValueError, match="joint_position_rad"):
        tick(adapter(), joints=np.full(7, np.nan))
    with pytest.raises(ValueError, match="joint positions"):
        tick(adapter(), joints=np.full(7, 1.1))
    bad = ActionPreview(np.zeros(7), np.ones(7), True)
    with pytest.raises(ValueError, match="does not match"):
        tick(adapter(), nominal=bad)
    with pytest.raises(ValueError, match="joint_bounds"):
        tick(adapter(), bounds=np.zeros((7, 1)))
    with pytest.raises(ValueError, match="early, late, or repeated"):
        tick(adapter(), 0.502)


def test_pi05_release_stops_without_residual_action():
    policy, wrench = MockResidual(), MockWrench()
    instance = adapter(policy, wrench)
    result = tick(instance, nominal=action(closed=False))
    assert result.release_requested and result.target_rad is None
    assert result.residual_delta_rad is None and not policy.observations
    assert wrench.calls == [0.5]
    with pytest.raises(RuntimeError, match="terminated"):
        tick(instance)


@pytest.mark.parametrize("force", [False, True])
def test_adapter_uses_real_simulated_sensor_across_control_ticks(force):
    env = PegInHoleEnv()
    env.reset(seed=0)
    sensor = ResidualWrenchInput(
        EndEffectorLoadCell(env), trial_seed=0, timestep_s=env.model.opt.timestep,
    )
    for _ in range(2):
        sensor.update(float(env.data.time))
        env.step()
    start = float(env.data.time)
    policy = MockResidual()
    instance = ResidualAdapter(
        policy, sensor, force_observation=force, start_time_s=start,
        physics_timestep_s=env.model.opt.timestep,
    )
    for i in range(3):
        tick_time = start + i / 15
        while float(env.data.time) + env.model.opt.timestep / 2 < tick_time:
            sensor.update(float(env.data.time))
            env.step()
        sensor.update(float(env.data.time))
        tick(instance, float(env.data.time))
        env.step()
    assert len(policy.observations) == 3
    for observation in policy.observations:
        assert isinstance(observation, ForceResidualObservation) == force
        if force:
            assert observation.wrench_time_s <= observation.time_s
            assert observation.time_s - observation.wrench_time_s == pytest.approx(0.004)
