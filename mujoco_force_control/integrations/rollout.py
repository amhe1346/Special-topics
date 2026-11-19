"""Guarded pi0.5 control for simulation only; not a hardware safety system."""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import asdict, dataclass
import time

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from envs.peg_episode import EpisodeConfig, PegEpisode
from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import CHUNK_SHAPE, DroidObservation, Policy, infer_preview
from integrations.residual_adapter import ResidualAdapter, ResidualPolicy
from integrations.residual_experiments import ExperimentConfig
from integrations.residual_reward import ForceSample, INVALID, TaskState, interval_reward
from integrations.sim_observation import capture_observation
from sensors.load_cell import EndEffectorLoadCell
from sensors.residual_wrench import ResidualWrenchInput, WrenchInputError


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


@dataclass(frozen=True)
class ResidualRollout:
    experiment: ExperimentConfig
    policy: ResidualPolicy
    trial_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentConfig):
            raise ValueError("experiment must be an ExperimentConfig")
        if type(self.trial_seed) is not int or self.trial_seed < 0:
            raise ValueError("trial_seed must be a non-negative integer")


def run_rollout(env: PegInHoleEnv, renderer: mujoco.Renderer, policy: Policy,
                config: RolloutConfig, *, prompt: str = "Insert the peg into the hole.",
                on_step: Callable[[PegInHoleEnv], None] | None = None,
                on_observation: Callable[[DroidObservation], None] | None = None,
                residual: ResidualRollout | None = None) -> dict:
    """Run a single closed-loop simulated episode, replanning at each action."""
    if residual is not None and (config.frequency_hz != 15.0 or config.max_joint_speed_rad_s != 0.1):
        raise ValueError("residual adapter requires 15 Hz and 0.1 rad/s nominal target limits")
    if residual is not None and config.no_contact:
        raise ValueError("residual reward rollout requires contact to be allowed")
    controller = PDPositionController(env)
    try:
        return _run_rollout(env, renderer, policy, config, controller, prompt, on_step, on_observation, residual)
    finally:
        env.data.ctrl[controller.actuator_ids] = 0.0


def _run_rollout(env: PegInHoleEnv, renderer: mujoco.Renderer, policy: Policy,
                 config: RolloutConfig, controller: PDPositionController, prompt: str,
                 on_step: Callable[[PegInHoleEnv], None] | None,
                 on_observation: Callable[[DroidObservation], None] | None,
                 residual: ResidualRollout | None) -> dict:
    held, _ = controller.read_joint_state()
    wrench = None
    sensor_env = None
    for settle_step in range(250):
        if residual is not None and settle_step >= 247:
            if sensor_env is None:
                sensor_env = copy.copy(env)
                sensor_env.data = mujoco.MjData(env.model)
            mujoco.mj_copyData(sensor_env.data, env.model, env.data)
            mujoco.mj_forward(env.model, sensor_env.data)
            if wrench is None:
                wrench = ResidualWrenchInput(
                    EndEffectorLoadCell(sensor_env), trial_seed=residual.trial_seed,
                    timestep_s=float(env.model.opt.timestep),
                )
            wrench.update(float(env.data.time))
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
    adapter: ResidualAdapter | None = None
    if residual is not None:
        assert wrench is not None
        wrench.source = EndEffectorLoadCell(env)
        adapter = residual.experiment.make_adapter(
            residual.policy, wrench, start_time_s=float(env.data.time), physics_timestep_s=timestep,
        )
    tick = 0
    durations = []
    decisions = []
    reason = None
    action_chunk = ()
    action_chunk_index = 0
    active_decision = None
    sensor_error = None
    reward_intervals = []
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
        joints, velocities = controller.read_joint_state()
        residual_result = None
        if adapter is not None:
            finger_positions = env.data.qpos[env.model.jnt_qposadr[env.finger_joint_ids]]
            width = float(np.sum(finger_positions))
            if not np.isfinite(width) or not -1e-5 <= width <= 0.08 + 1e-5:
                raise ValueError("simulated gripper width is outside its expected range")
            try:
                residual_result = adapter.step(
                    time_s=float(env.data.time), joint_position_rad=joints,
                    joint_velocity_rad_s=velocities,
                    gripper_closed_fraction=float(np.clip(1.0 - width / 0.08, 0.0, 1.0)),
                    nominal=action, joint_bounds_rad=bounds,
                )
            except WrenchInputError as error:
                reason, sensor_error = "wrench_input_error", str(error)
                break
            if residual_result.release_requested:
                reason = "release_requested"
                geometry_at_release = env.measure_geometry()
                state = TaskState(geometry_at_release.lateral_error, geometry_at_release.depth)
                breakdown = interval_reward(
                    state, state, [], force_aware=residual.experiment.force_aware_reward,
                    termination=reason,
                )
                reward_intervals.append({
                    "kind": "nominal_release_event", "start_time_s": float(env.data.time),
                    "end_time_s": float(env.data.time), "physics_steps": 0,
                    "termination": reason, "breakdown": asdict(breakdown),
                })
                break
            assert residual_result.target_rad is not None
            target = residual_result.target_rad
        else:
            if not action.gripper_closed:
                reason = "release_requested"
                break
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
        if residual_result is not None:
            assert residual_result.residual_raw_rad is not None
            assert residual_result.residual_delta_rad is not None
            assert residual_result.residual_clipped is not None
            assert residual_result.total_delta_clipped is not None
            assert residual_result.joint_limit_clipped is not None
            execution["residual"] = {
                "wrench_time_s": residual_result.wrench_time_s,
                "wrench_local": residual_result.wrench_local.tolist(),
                "nominal_delta_rad": residual_result.nominal_delta_rad.tolist(),
                "raw_delta_rad": residual_result.residual_raw_rad.tolist(),
                "bounded_delta_rad": residual_result.residual_delta_rad.tolist(),
                "residual_clipped": residual_result.residual_clipped.tolist(),
                "total_delta_clipped": residual_result.total_delta_clipped.tolist(),
                "joint_limit_clipped": residual_result.joint_limit_clipped.tolist(),
            }
        active_decision["executed_actions"].append(execution)
        if len(active_decision["executed_actions"]) == 1:
            active_decision["joint_target_rad"] = target.tolist()
        tick += 1
        end_time = episode.start_time + tick / config.frequency_hz
        samples_at_tick_start = len(episode.samples)
        if residual is not None:
            initial_geometry = env.measure_geometry()
            initial_state = TaskState(initial_geometry.lateral_error, initial_geometry.depth)
            interval_start_time = float(env.data.time)
        while env.data.time + timestep / 2 < end_time:
            if wrench is not None:
                try:
                    wrench.update(float(env.data.time))
                except WrenchInputError as error:
                    reason, sensor_error = "wrench_input_error", str(error)
                    if len(episode.samples) == samples_at_tick_start:
                        active_decision["executed_actions"].pop()
                        if not active_decision["executed_actions"]:
                            active_decision.pop("joint_target_rad", None)
                    break
            controller.step(target)
            reason = episode.step()
            if on_step is not None and reason != "invalid_state":
                on_step(env)
            if reason:
                break
            if config.no_contact and episode.samples[-1].force_n > 0:
                reason = "contact"
                break
        if residual is not None and reason not in INVALID:
            final_geometry = env.measure_geometry()
            final_state = TaskState(final_geometry.lateral_error, final_geometry.depth)
            interval_samples = episode.samples[samples_at_tick_start:]
            if not interval_samples:
                raise RuntimeError("residual action produced no physics samples")
            breakdown = interval_reward(
                initial_state, final_state,
                [ForceSample(sample.force_n, timestep) for sample in interval_samples],
                force_aware=residual.experiment.force_aware_reward, termination=reason,
            )
            execution["reward"] = asdict(breakdown)
            reward_intervals.append({
                "kind": "residual_action", "start_time_s": interval_start_time,
                "end_time_s": float(env.data.time), "physics_steps": len(interval_samples),
                "termination": reason, "breakdown": asdict(breakdown),
            })
    if reason in ("inference_timeout", "stale_observation", "release_requested", "contact", "wrench_input_error"):
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
    result = {
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
    if residual is not None:
        result["residual_experiment"] = {
            "variant": residual.experiment.variant.value,
            "force_observation": residual.experiment.force_observation,
            "force_aware_reward": residual.experiment.force_aware_reward,
            "reward_implemented": True,
            "reward_status": "invalid" if reason in INVALID else "complete",
            "reward_intervals": reward_intervals,
            "undiscounted_reward_total": (
                None if reason in INVALID else sum(item["breakdown"]["total"] for item in reward_intervals)
            ),
            "training_implemented": False,
            "wrench_seed": residual.trial_seed,
            "wrench_tare_local": wrench.tare_local.tolist(),
            "sensor_error": sensor_error,
        }
    return result
