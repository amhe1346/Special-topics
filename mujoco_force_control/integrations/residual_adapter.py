"""Simulation-only, per-tick residual adapter; does not execute controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from integrations.pi05_droid import ActionPreview, droid_joint_velocity_to_delta
from sensors.residual_wrench import WrenchInputError, WrenchSample


class WrenchInput(Protocol):
    def sample(self, time_s: float) -> WrenchSample:
        """Return the latest healthy sensor sample."""


@dataclass(frozen=True)
class ResidualObservation:
    time_s: float
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    gripper_closed_fraction: float
    nominal_delta_rad: np.ndarray
    nominal_gripper_closed: bool


@dataclass(frozen=True)
class ForceResidualObservation(ResidualObservation):
    wrench_time_s: float
    wrench_local: np.ndarray


class ResidualPolicy(Protocol):
    def infer(self, observation: ResidualObservation) -> np.ndarray:
        """Return seven joint corrections in radians per tick."""


@dataclass(frozen=True)
class AdapterResult:
    time_s: float
    wrench_time_s: float
    wrench_local: np.ndarray
    release_requested: bool
    nominal_delta_rad: np.ndarray
    residual_raw_rad: np.ndarray | None
    residual_delta_rad: np.ndarray | None
    residual_clipped: np.ndarray | None
    total_delta_clipped: np.ndarray | None
    joint_limit_clipped: np.ndarray | None
    target_rad: np.ndarray | None
    executed_delta_rad: np.ndarray | None


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


class ResidualAdapter:
    """Apply one bounded residual at each 15 Hz simulation control tick."""

    def __init__(
        self, policy: ResidualPolicy, wrench: WrenchInput, *, force_observation: bool,
        start_time_s: float, physics_timestep_s: float,
    ) -> None:
        if type(force_observation) is not bool:
            raise ValueError("force_observation must be a bool")
        if not np.isfinite(start_time_s) or start_time_s < 0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not np.isfinite(physics_timestep_s) or not 0 < physics_timestep_s < 1 / 15:
            raise ValueError("physics_timestep_s must be positive and less than one control period")
        self.policy = policy
        self.wrench = wrench
        self.force_observation = force_observation
        self.start_time_s = float(start_time_s)
        self.physics_timestep_s = float(physics_timestep_s)
        self._tick = 0
        self._terminated = False

    def step(
        self, *, time_s: float, joint_position_rad: np.ndarray,
        joint_velocity_rad_s: np.ndarray, gripper_closed_fraction: float,
        nominal: ActionPreview, joint_bounds_rad: np.ndarray,
    ) -> AdapterResult:
        """Return a target or release request; callers must stop on errors."""
        if self._terminated:
            raise RuntimeError("adapter has already terminated")
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("time_s must be finite and non-negative")
        expected = self.start_time_s + self._tick / 15.0
        if abs(time_s - expected) > self.physics_timestep_s / 2 + 1e-9:
            raise ValueError("control tick is early, late, or repeated")
        joints = _vector(joint_position_rad, 7, "joint_position_rad")
        velocity = _vector(joint_velocity_rad_s, 7, "joint_velocity_rad_s")
        if not np.isfinite(gripper_closed_fraction) or not 0 <= gripper_closed_fraction <= 1:
            raise ValueError("gripper_closed_fraction must be between 0 and 1")
        bounds = np.asarray(joint_bounds_rad, dtype=np.float64)
        if bounds.shape != (7, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("joint_bounds_rad must contain seven valid finite ranges")
        if np.any(joints < bounds[:, 0]) or np.any(joints > bounds[:, 1]):
            raise ValueError("joint positions must be inside joint bounds")
        nominal_velocity = _vector(nominal.normalized_velocity, 7, "nominal velocity")
        nominal_delta = _vector(nominal.joint_delta_rad, 7, "nominal joint delta")
        if not np.allclose(nominal_delta, droid_joint_velocity_to_delta(nominal_velocity), rtol=0, atol=1e-12):
            raise ValueError("nominal joint delta does not match normalized velocity")
        if type(nominal.gripper_closed) is not bool:
            raise ValueError("nominal gripper state must be a bool")

        sample = self.wrench.sample(float(time_s))
        if not np.isfinite(sample.time_s) or sample.time_s < 0 or sample.time_s > time_s:
            raise WrenchInputError("invalid or future-dated wrist wrench")
        if time_s - sample.time_s > 0.1:
            raise WrenchInputError("wrist wrench is stale")
        try:
            wrench = _vector(sample.wrench_local, 6, "wrist wrench")
        except ValueError as error:
            raise WrenchInputError(str(error)) from error

        nominal_bounded = np.clip(nominal_delta, -0.1 / 15, 0.1 / 15)
        if not nominal.gripper_closed:
            self._terminated = True
            return AdapterResult(
                float(time_s), float(sample.time_s), wrench.copy(), True, nominal_bounded,
                None, None, None, None, None, None, None,
            )
        common = dict(
            time_s=float(time_s), joint_position_rad=joints.copy(),
            joint_velocity_rad_s=velocity, gripper_closed_fraction=float(gripper_closed_fraction),
            nominal_delta_rad=nominal_bounded.copy(), nominal_gripper_closed=True,
        )
        observation = (
            ForceResidualObservation(**common, wrench_time_s=float(sample.time_s), wrench_local=wrench)
            if self.force_observation else ResidualObservation(**common)
        )
        raw = _vector(self.policy.infer(observation), 7, "residual output")
        residual = np.clip(raw, -0.001, 0.001)
        combined = nominal_bounded + residual
        bounded = np.clip(combined, -0.1 / 15, 0.1 / 15)
        unbounded_target = joints + bounded
        target = np.clip(unbounded_target, bounds[:, 0], bounds[:, 1])
        self._tick += 1
        return AdapterResult(
            float(time_s), float(sample.time_s), wrench.copy(), False, nominal_bounded, raw, residual,
            raw != residual, combined != bounded, unbounded_target != target,
            target, target - joints,
        )
