"""Guarded pi0.5 control for simulation only; not a hardware safety system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from envs.peg_episode import EpisodeConfig, PegEpisode
from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import DroidObservation, Policy, infer_preview
from integrations.sim_observation import capture_observation


@dataclass(frozen=True)
class RolloutConfig:
    timeout_s: float = 6.0
    max_joint_speed_rad_s: float = 0.1
    max_inference_s: float = 5.0
    frequency_hz: float = 15.0
    no_contact: bool = True

    def __post_init__(self) -> None:
        for name in ("timeout_s", "max_joint_speed_rad_s", "max_inference_s", "frequency_hz"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def run_rollout(env: PegInHoleEnv, renderer: mujoco.Renderer, policy: Policy,
                config: RolloutConfig, *, prompt: str = "Insert the peg into the hole.",
                on_step: Callable[[PegInHoleEnv], None] | None = None,
                on_observation: Callable[[DroidObservation], None] | None = None) -> dict:
    """Run a single closed-loop simulated episode, replanning at each action."""
    controller = PDPositionController(env)
    try:
        return _run_rollout(env, renderer, policy, config, controller, prompt, on_step, on_observation)
    finally:
        env.data.ctrl[controller.actuator_ids] = 0.0


def _run_rollout(env: PegInHoleEnv, renderer: mujoco.Renderer, policy: Policy,
                 config: RolloutConfig, controller: PDPositionController, prompt: str,
                 on_step: Callable[[PegInHoleEnv], None] | None,
                 on_observation: Callable[[DroidObservation], None] | None) -> dict:
    held, _ = controller.read_joint_state()
    for _ in range(250):
        controller.step(held)
        env.step()
    mujoco.mj_forward(env.model, env.data)
    episode = PegEpisode(env, EpisodeConfig(timeout_s=config.timeout_s))
    if not episode._touching_finger():
        raise RuntimeError("Initial peg grasp is not established")
    timestep = float(env.model.opt.timestep)
    if config.frequency_hz * timestep > 1:
        raise ValueError("control frequency cannot exceed physics frequency")
    bounds = env.model.jnt_range[controller.joint_ids]
    tick = 0
    durations = []
    decisions = []
    reason = None
    initial_fixture = env.data.mocap_pos[env.fixture_mocap_id].copy()
    while reason is None:
        observation = capture_observation(env, renderer, prompt=prompt)
        if on_observation is not None:
            on_observation(observation)
        start = time.monotonic()
        try:
            actions = infer_preview(policy, observation)
        except TimeoutError:
            durations.append(time.monotonic() - start)
            reason = "inference_timeout"
            break
        elapsed = time.monotonic() - start
        durations.append(elapsed)
        if elapsed > config.max_inference_s:
            reason = "inference_timeout"
            break
        if float(env.data.time) != observation.time_s:
            reason = "stale_observation"
            break
        action = actions[0]
        decisions.append({
            "observation_time_s": observation.time_s,
            "inference_wall_s": elapsed,
            "normalized_velocity": action.normalized_velocity.tolist(),
            "gripper_closed": action.gripper_closed,
        })
        if not action.gripper_closed:
            reason = "release_requested"
            break
        joints, _ = controller.read_joint_state()
        delta = np.clip(action.joint_delta_rad, -config.max_joint_speed_rad_s / config.frequency_hz,
                        config.max_joint_speed_rad_s / config.frequency_hz)
        target = np.clip(joints + delta, bounds[:, 0], bounds[:, 1])
        decisions[-1]["joint_target_rad"] = target.tolist()
        tick += 1
        end_time = episode.start_time + tick / config.frequency_hz
        while env.data.time + timestep / 2 < end_time:
            controller.step(target)
            reason = episode.step()
            if on_step is not None and reason != "invalid_state":
                on_step(env)
            if reason:
                break
            if config.no_contact and episode.samples[-1].force_n > 0:
                reason = "contact"
                break
    if reason in ("inference_timeout", "stale_observation", "release_requested", "contact"):
        episode._finish_interval(float(env.data.time))
    geometry = None if reason == "invalid_state" else env.measure_geometry()
    sim_time = float(env.data.time - episode.start_time)
    return {
        "reason": reason, "sim_time_s": sim_time if np.isfinite(sim_time) else None,
        "inference_s": durations, "steps": len(episode.samples),
        "decisions": decisions,
        "peak_force_n": episode.peak_force_n,
        "depth_m": None if geometry is None else geometry.depth,
        "fixture_unchanged": bool(np.array_equal(initial_fixture, env.data.mocap_pos[env.fixture_mocap_id])),
    }
