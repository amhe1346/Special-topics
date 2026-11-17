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
from integrations.pi05_droid import CHUNK_SHAPE, DroidObservation, Policy, infer_preview
from integrations.sim_observation import capture_observation


def _inference_timing_summary(
    durations: list[float], frequency_hz: float, open_loop_horizon: int = 1
) -> dict:
    """Summarize wall-clock inference against its action-chunk budget."""
    period = 1.0 / frequency_hz
    budget = period * open_loop_horizon
    if not durations:
        return {
            "control_period_s": period,
            "inference_budget_s": budget,
            "open_loop_horizon": open_loop_horizon,
            "calls": 0,
            "p50_s": None,
            "p95_s": None,
            "max_s": None,
            "deadline_misses": 0,
            "deadline_miss_rate": None,
        }
    values = np.asarray(durations, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("inference durations must be finite and non-negative")
    misses = int(np.count_nonzero(values > budget))
    return {
        "control_period_s": period,
        "inference_budget_s": budget,
        "open_loop_horizon": open_loop_horizon,
        "calls": int(values.size),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "max_s": float(np.max(values)),
        "deadline_misses": misses,
        "deadline_miss_rate": float(misses / values.size),
    }


@dataclass(frozen=True)
class RolloutConfig:
    timeout_s: float = 6.0
    max_joint_speed_rad_s: float = 0.1
    max_inference_s: float = 5.0
    frequency_hz: float = 15.0
    open_loop_horizon: int = 8
    no_contact: bool = True

    def __post_init__(self) -> None:
        for name in ("timeout_s", "max_joint_speed_rad_s", "max_inference_s", "frequency_hz"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not isinstance(self.open_loop_horizon, int)
                or not 1 <= self.open_loop_horizon <= CHUNK_SHAPE[0]):
            raise ValueError(f"open_loop_horizon must be between 1 and {CHUNK_SHAPE[0]}")


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
    action_chunk = ()
    action_chunk_index = 0
    active_decision = None
    initial_fixture = env.data.mocap_pos[env.fixture_mocap_id].copy()
    while reason is None:
        if action_chunk_index >= len(action_chunk):
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
            action_chunk = actions[:config.open_loop_horizon]
            action_chunk_index = 0
            first_action = action_chunk[0]
            active_decision = {
                "observation_time_s": observation.time_s,
                "inference_wall_s": elapsed,
                "normalized_velocity": first_action.normalized_velocity.tolist(),
                "gripper_closed": first_action.gripper_closed,
                "executed_actions": [],
            }
            decisions.append(active_decision)
        assert active_decision is not None
        action = action_chunk[action_chunk_index]
        action_chunk_index += 1
        if not action.gripper_closed:
            reason = "release_requested"
            break
        joints, _ = controller.read_joint_state()
        delta = np.clip(action.joint_delta_rad, -config.max_joint_speed_rad_s / config.frequency_hz,
                        config.max_joint_speed_rad_s / config.frequency_hz)
        target = np.clip(joints + delta, bounds[:, 0], bounds[:, 1])
        executed_delta = target - joints
        executed_velocity = executed_delta / 0.2
        execution = {
            "normalized_velocity": executed_velocity.tolist(),
            "joint_delta_rad": executed_delta.tolist(),
            "joint_target_rad": target.tolist(),
            "gripper_closed": action.gripper_closed,
            "simulation_time_s": float(env.data.time),
        }
        active_decision["executed_actions"].append(execution)
        if len(active_decision["executed_actions"]) == 1:
            active_decision["joint_target_rad"] = target.tolist()
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
    forces = np.asarray([sample.force_n for sample in episode.samples], dtype=float)
    force_summary = {
        "measurement": "sum of peg-fixture and peg-table contact force magnitudes",
        "sample_count": len(forces),
        "mean_n": float(np.mean(forces)) if len(forces) else None,
        "p95_n": float(np.percentile(forces, 95)) if len(forces) else None,
        "peak_n": float(episode.peak_force_n),
        "reference_threshold_n": episode.config.reference_force_n,
        "time_above_reference_s": float(sum(
            end - start for start, end in episode.above_reference_intervals
        )),
        "hard_limit_n": episode.config.hard_force_n,
    }
    return {
        "reason": reason, "sim_time_s": sim_time if np.isfinite(sim_time) else None,
        "inference_s": durations, "steps": len(episode.samples),
        "action_steps": sum(len(item["executed_actions"]) for item in decisions),
        "open_loop_horizon": config.open_loop_horizon,
        "inference_timing": _inference_timing_summary(
            durations, config.frequency_hz, config.open_loop_horizon
        ),
        "decisions": decisions,
        "peak_force_n": episode.peak_force_n,
        "force_summary": force_summary,
        "depth_m": None if geometry is None else geometry.depth,
        "fixture_unchanged": bool(np.array_equal(initial_fixture, env.data.mocap_pos[env.fixture_mocap_id])),
    }
