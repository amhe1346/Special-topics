from dataclasses import FrozenInstanceError, asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from integrations.pi05_droid import preview_actions
from integrations.residual_adapter import ForceResidualObservation, ResidualObservation
from integrations.residual_experiments import CONFIGURATIONS, ExperimentConfig, Variant, get_config
from sensors.residual_wrench import WrenchInputError, WrenchSample


class Policy:
    def __init__(self):
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        return np.zeros(7)


class Wrench:
    def __init__(self, *, stale=False):
        self.calls = 0
        self.stale = stale

    def sample(self, time_s):
        self.calls += 1
        return WrenchSample(time_s - 0.2 if self.stale else time_s, np.ones(6))


def run_tick(adapter):
    action = np.tile(np.r_[np.zeros(7), 1.0], (15, 1))
    return adapter.step(
        time_s=1.0, joint_position_rad=np.zeros(7), joint_velocity_rad_s=np.zeros(7),
        gripper_closed_fraction=1.0, nominal=preview_actions({"actions": action})[0],
        joint_bounds_rad=np.tile([-1.0, 1.0], (7, 1)),
    )


def test_complete_unique_factorial_matrix_and_immutable_configs():
    assert len(CONFIGURATIONS) == 4
    assert {c.variant for c in CONFIGURATIONS} == set(Variant)
    assert {(c.force_observation, c.force_aware_reward) for c in CONFIGURATIONS} == {
        (False, False), (True, False), (False, True), (True, True),
    }
    for config in CONFIGURATIONS:
        assert get_config(config.variant) is config
        with pytest.raises(FrozenInstanceError):
            config.force_observation = not config.force_observation


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=lambda c: c.variant.value)
def test_variant_constructs_correct_adapter_observation_and_shared_health_gate(config):
    policy, wrench = Policy(), Wrench()
    adapter = config.make_adapter(policy, wrench, start_time_s=1, physics_timestep_s=0.002)
    result = run_tick(adapter)
    assert not result.release_requested
    assert wrench.calls == 1
    assert type(policy.observations[0]) is (
        ForceResidualObservation if config.force_observation else ResidualObservation
    )
    if not config.force_observation:
        assert not hasattr(policy.observations[0], "wrench_local")
        assert not hasattr(policy.observations[0], "wrench_time_s")

    policy, wrench = Policy(), Wrench(stale=True)
    adapter = config.make_adapter(policy, wrench, start_time_s=1, physics_timestep_s=0.002)
    with pytest.raises(WrenchInputError, match="stale"):
        run_tick(adapter)
    assert wrench.calls == 1
    assert not policy.observations


def test_inconsistent_or_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="factors"):
        ExperimentConfig(Variant.BASELINE, True, False)
    with pytest.raises(ValueError, match="factors"):
        ExperimentConfig(Variant.BASELINE, 0, False)
    with pytest.raises(ValueError, match="variant"):
        get_config("baseline")


def test_cli_outputs_all_configs_and_selects_exact_variant():
    command = [sys.executable, "-m", "integrations.residual_experiments"]
    all_configs = json.loads(subprocess.check_output(command, text=True))
    assert all_configs["status"] == "design_only"
    assert all_configs["training_implemented"] is False
    assert all_configs["reward_implemented"] is True
    assert all_configs["reward_scope"] == "simulation_rollout_pilot"
    assert all_configs["configurations"] == [
        {**asdict(config), "variant": config.variant.value} for config in CONFIGURATIONS
    ]
    one = json.loads(subprocess.check_output(command + ["--variant", "reward_only"], text=True))
    assert one["configurations"] == [
        {"variant": "reward_only", "force_observation": False, "force_aware_reward": True}
    ]
    invalid = subprocess.run(command + ["--variant", "unknown"], capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "invalid choice" in invalid.stderr
