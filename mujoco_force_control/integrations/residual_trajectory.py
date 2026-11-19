"""Observation and episode contracts for exploratory residual training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from integrations.residual_adapter import ForceResidualObservation, ResidualObservation
from integrations.residual_reward import FAILURES, INVALID


def observation_vector(observation: ResidualObservation, *, force_observation: bool) -> np.ndarray:
    """Encode physical inputs with fixed scales; no training statistics or clipping."""
    if type(force_observation) is not bool:
        raise ValueError("force_observation must be a bool")
    if force_observation != isinstance(observation, ForceResidualObservation):
        raise ValueError("observation type does not match force variant")
    if observation.nominal_gripper_closed is not True:
        raise ValueError("release requests are not residual observations")

    def vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result

    grip = float(observation.gripper_closed_fraction)
    if not math.isfinite(grip) or not 0 <= grip <= 1:
        raise ValueError("gripper fraction must be in [0, 1]")
    pieces = [
        vector(observation.joint_position_rad, 7, "joint position") / math.pi,
        vector(observation.joint_velocity_rad_s, 7, "joint velocity"),
        np.array([grip]),
        vector(observation.nominal_delta_rad, 7, "nominal delta") / (0.1 / 15),
    ]
    if isinstance(observation, ForceResidualObservation):
        wrench = vector(observation.wrench_local, 6, "wrench")
        pieces.append(wrench / np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]))
    return np.concatenate(pieces)


@dataclass(frozen=True)
class EpisodeRewards:
    rewards: np.ndarray
    terminal_reason: str
    unassigned_release_reward: float


def episode_rewards(result: dict[str, Any], action_count: int) -> EpisodeRewards:
    """Map rollout intervals to policy actions, rejecting all invalid episodes."""
    if type(action_count) is not int or action_count < 0:
        raise ValueError("action_count must be a non-negative integer")
    reason = result["reason"]
    experiment = result["residual_experiment"]
    if reason in INVALID or experiment["reward_status"] != "complete":
        detail = experiment.get("sensor_error")
        raise RuntimeError(f"invalid rollout; abort training: {reason}" + (f": {detail}" if detail else ""))
    if reason != "success" and reason not in FAILURES:
        raise ValueError(f"unexpected rollout termination: {reason}")
    intervals = experiment["reward_intervals"]
    actions = [item for item in intervals if item["kind"] == "residual_action"]
    releases = [item for item in intervals if item["kind"] == "nominal_release_event"]
    if len(actions) + len(releases) != len(intervals) or len(actions) != action_count:
        raise ValueError("rollout action and reward interval counts do not match")
    if result["action_steps"] != action_count:
        raise ValueError("rollout action count does not match policy calls")
    if len(releases) != (1 if reason == "release_requested" else 0):
        raise ValueError("release event does not match termination")
    if releases and (intervals[-1] is not releases[0] or releases[0]["termination"] != reason):
        raise ValueError("release must be the final event")
    if any(item["termination"] is not None for item in actions[:-1]):
        raise ValueError("only the final action may terminate")
    expected_action_terminal = None if releases else reason
    if actions and actions[-1]["termination"] != expected_action_terminal:
        raise ValueError("final action termination does not match rollout")
    if not actions and not releases:
        raise ValueError("terminal rollout has no actions or release event")
    rewards = np.asarray([item["breakdown"]["total"] for item in actions], dtype=np.float64)
    unassigned = 0.0
    if releases:
        release_reward = float(releases[0]["breakdown"]["total"])
        if not math.isfinite(release_reward) or release_reward != -5.0:
            raise ValueError("invalid nominal release reward")
        if action_count:
            rewards[-1] += release_reward
        else:
            unassigned = release_reward
    if not np.all(np.isfinite(rewards)):
        raise ValueError("non-finite episode rewards")
    total = float(experiment["undiscounted_reward_total"])
    if not math.isfinite(total) or not math.isclose(float(rewards.sum()) + unassigned, total, abs_tol=1e-9):
        raise ValueError("episode rewards do not match rollout total")
    return EpisodeRewards(rewards, reason, unassigned)
