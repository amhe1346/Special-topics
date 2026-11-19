import copy

import numpy as np
import pytest

from envs.peg_in_hole import PegInHoleEnv
from integrations.residual_adapter import ForceResidualObservation, ResidualObservation
from integrations.residual_experiments import Variant, get_config
from integrations.residual_trajectory import episode_rewards, observation_vector
from integrations.rollout import ResidualRollout, RolloutConfig, run_rollout
from tests.test_residual_rollout import Nominal, Renderer, Residual


@pytest.mark.parametrize("variant", list(Variant))
def test_real_rollout_observations_and_rewards(variant):
    env = PegInHoleEnv()
    env.reset(seed=0, lateral_range=0.00005)
    policy = Residual()
    experiment = get_config(variant)
    result = run_rollout(
        env, Renderer(), Nominal(), RolloutConfig(timeout_s=0.1, no_contact=False),
        residual=ResidualRollout(experiment, policy, 0),
    )
    rewards = episode_rewards(result, len(policy.observations))
    assert rewards.terminal_reason == "timeout"
    assert len(rewards.rewards) == 2
    assert rewards.rewards[-1] < -4
    assert sum(rewards.rewards) == pytest.approx(result["residual_experiment"]["undiscounted_reward_total"])
    for observation in policy.observations:
        encoded = observation_vector(observation, force_observation=experiment.force_observation)
        assert encoded.shape == (28 if experiment.force_observation else 22,)
        with pytest.raises(ValueError, match="observation type"):
            observation_vector(observation, force_observation=not experiment.force_observation)


def test_observation_scales_and_no_wrench_leak():
    common = dict(
        time_s=3.0, joint_position_rad=np.full(7, np.pi),
        joint_velocity_rad_s=np.full(7, 2.0), gripper_closed_fraction=0.5,
        nominal_delta_rad=np.full(7, 0.1 / 15), nominal_gripper_closed=True,
    )
    plain = observation_vector(ResidualObservation(**common), force_observation=False)
    force = observation_vector(
        ForceResidualObservation(**common, wrench_time_s=2.996,
                                 wrench_local=np.array([5, 10, 15, 1, 2, 3])),
        force_observation=True,
    )
    np.testing.assert_array_equal(plain[:7], 1)
    np.testing.assert_array_equal(plain[7:14], 2)
    assert plain[14] == 0.5
    np.testing.assert_array_equal(plain[15:], 1)
    np.testing.assert_array_equal(force[:22], plain)
    np.testing.assert_array_equal(force[22:], [1, 2, 3, 1, 2, 3])


@pytest.mark.parametrize("closed_actions", [0, 1])
def test_real_rollout_release_mapping(closed_actions):
    class Release:
        def infer(self, observation):
            actions = np.zeros((15, 8))
            actions[:closed_actions, 7] = 1
            return {"actions": actions}

    env = PegInHoleEnv()
    env.reset(seed=0, lateral_range=0.00005)
    policy = Residual()
    result = run_rollout(
        env, Renderer(), Release(), RolloutConfig(timeout_s=0.3, no_contact=False),
        residual=ResidualRollout(get_config(Variant.BASELINE), policy, 0),
    )
    mapped = episode_rewards(result, len(policy.observations))
    assert mapped.terminal_reason == "release_requested"
    assert len(mapped.rewards) == closed_actions
    assert mapped.unassigned_release_reward == (-5 if not closed_actions else 0)
    if closed_actions:
        intervals = result["residual_experiment"]["reward_intervals"]
        assert mapped.rewards[-1] == pytest.approx(intervals[0]["breakdown"]["total"] - 5)
    assert sum(mapped.rewards) + mapped.unassigned_release_reward == pytest.approx(
        result["residual_experiment"]["undiscounted_reward_total"]
    )


def test_reject_invalid_episode_and_inconsistent_count():
    result = {
        "reason": "wrench_input_error", "action_steps": 0,
        "residual_experiment": {"reward_status": "invalid"},
    }
    with pytest.raises(RuntimeError, match="abort training"):
        episode_rewards(result, 0)
    result = {
        "reason": "timeout", "action_steps": 1,
        "residual_experiment": {
            "reward_status": "complete", "undiscounted_reward_total": -5,
            "reward_intervals": [{"kind": "residual_action", "termination": "timeout",
                                  "breakdown": {"total": -5}}],
        },
    }
    with pytest.raises(ValueError, match="counts"):
        episode_rewards(copy.deepcopy(result), 0)
