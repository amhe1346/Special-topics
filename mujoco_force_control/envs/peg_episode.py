"""Simulation-only termination, seating, and contact-force accounting."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from envs.peg_contact import PegContactForce
from envs.peg_in_hole import GEOMETRY_ROUNDOFF, HOLE_RADIUS, PEG_RADIUS, PegInHoleEnv


@dataclass(frozen=True)
class EpisodeConfig:
    dwell_s: float = 0.1
    angle_rad: float = float(np.deg2rad(0.2))
    reference_force_n: float = 1.0
    hard_force_n: float = 5.0
    timeout_s: float = 20.0
    workspace_xy_m: float = 0.3
    workspace_below_m: float = 0.05
    workspace_above_m: float = 0.3
    drop_distance_m: float = 0.02
    grasp_loss_s: float = 0.1

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Episode configuration values must be finite and positive")
        if self.reference_force_n >= self.hard_force_n:
            raise ValueError("reference_force_n must be lower than hard_force_n")
        if self.angle_rad >= np.pi / 2:
            raise ValueError("angle_rad must be less than pi/2")


@dataclass(frozen=True)
class EpisodeSample:
    time_s: float
    depth_m: float
    force_n: float
    force_world_n: tuple[float, float, float]
    seated: bool


class PegEpisode:
    """Evaluate every simulation step; construct a new episode after each reset."""

    def __init__(self, env: PegInHoleEnv, config: EpisodeConfig | None = None) -> None:
        self.env = env
        self.config = config or EpisodeConfig()
        self.contact = PegContactForce(env)
        self.finger_bodies = {
            env._named_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_finger", "right_finger")
        }
        self.start_time = float(env.data.time)
        self.origin = env.data.site_xpos[env.hole_site_id].copy()
        self.initial_hand_offset = self._hand_offset()
        self.release_intended = False
        self.reason: str | None = None
        self.dwell_s = 0.0
        self.peak_force_n = 0.0
        self.samples: list[EpisodeSample] = []
        self.above_reference_intervals: list[tuple[float, float]] = []
        self._above_start: float | None = None
        self._no_grasp_s = 0.0

    def _hand_offset(self) -> np.ndarray:
        data = self.env.data
        position = data.qpos[self.env.peg_qpos_address : self.env.peg_qpos_address + 3]
        return data.xmat[self.env.hand_id].reshape(3, 3).T @ (position - data.xpos[self.env.hand_id])

    def set_release_intent(self, intended: bool) -> None:
        """Explicitly permit a release; opening the gripper alone is not intent."""
        if not isinstance(intended, bool):
            raise TypeError("intended must be a bool")
        self.release_intended = intended

    def _seated(self) -> bool:
        env = self.env
        geometry = env.measure_geometry()
        if not geometry.geometric_candidate or geometry.angular_error > self.config.angle_rad:
            return False
        rotation = env.data.site_xmat[env.hole_site_id].reshape(3, 3)
        tip = rotation.T @ (env.data.site_xpos[env.peg_tip_id] - env.data.site_xpos[env.hole_site_id])
        axis = rotation.T @ (env.data.site_xpos[env.peg_tip_id] - env.data.site_xpos[env.peg_base_id])
        entrance = tip - axis * (tip[2] / axis[2])
        max_offset = max(np.linalg.norm(tip[:2]), np.linalg.norm(entrance[:2]))
        projected_radius = PEG_RADIUS / np.cos(geometry.angular_error)
        return bool(max_offset + projected_radius <= HOLE_RADIUS + GEOMETRY_ROUNDOFF)

    def _touching_finger(self) -> bool:
        for contact in self.env.data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0:
                continue
            if geom1 == self.contact.peg_geom and int(self.env.model.geom_bodyid[geom2]) in self.finger_bodies:
                return True
            if geom2 == self.contact.peg_geom and int(self.env.model.geom_bodyid[geom1]) in self.finger_bodies:
                return True
        return False

    def _finish_interval(self, end: float) -> None:
        if self._above_start is not None:
            self.above_reference_intervals.append((self._above_start, end))
            self._above_start = None

    def step(self) -> str | None:
        """Advance one step and return a persistent termination reason, if any."""
        if self.reason is not None:
            raise RuntimeError("Episode has terminated; reset before stepping again")
        env = self.env
        dt = float(env.model.opt.timestep)
        before = float(env.data.time)
        expected = self.start_time + len(self.samples) * dt
        if not np.isfinite(before) or abs(before - expected) > 1e-8:
            self.reason = "invalid_state"
            self._finish_interval(before if np.isfinite(before) else expected)
            return self.reason
        warnings = np.array([warning.number for warning in env.data.warning])
        if not all(np.all(np.isfinite(value)) for value in (
            env.data.qpos, env.data.qvel, env.data.ctrl, env.data.sensordata,
            env.data.mocap_pos, env.data.mocap_quat,
        )):
            self.reason = "invalid_state"
            self._finish_interval(before)
            return self.reason
        env.step()
        if (not np.isfinite(env.data.time) or abs(env.data.time - before - dt) > 1e-8
                or np.any(np.array([warning.number for warning in env.data.warning]) > warnings)
                or not all(np.all(np.isfinite(value)) for value in (env.data.qpos, env.data.qvel, env.data.sensordata))):
            self.reason = "invalid_state"
            self._finish_interval(before)
            return self.reason
        mujoco.mj_forward(env.model, env.data)
        if (np.any(np.array([warning.number for warning in env.data.warning]) > warnings)
                or not all(np.all(np.isfinite(value)) for value in (
                    env.data.qpos, env.data.qvel, env.data.sensordata,
                    env.data.site_xpos, env.data.site_xmat,
                ))):
            self.reason = "invalid_state"
            self._finish_interval(before)
            return self.reason
        try:
            force = self.contact.measure()
        except ValueError:
            self.reason = "invalid_state"
            self._finish_interval(before)
            return self.reason
        now = float(env.data.time)
        self.peak_force_n = max(self.peak_force_n, force.magnitude_sum)
        if force.magnitude_sum > self.config.reference_force_n:
            if self._above_start is None:
                self._above_start = before
        else:
            self._finish_interval(before)
        seated = self._seated()
        self.dwell_s = self.dwell_s + dt if seated else 0.0
        self._no_grasp_s = 0.0 if self.release_intended or self._touching_finger() else self._no_grasp_s + dt
        self.samples.append(EpisodeSample(now, env.measure_geometry().depth, force.magnitude_sum, force.world_vector, seated))
        tip = env.data.site_xpos[env.peg_tip_id]
        delta = tip - self.origin
        if force.magnitude_sum > self.config.hard_force_n:
            self.reason = "force_limit"
        elif (np.any(np.abs(delta[:2]) > self.config.workspace_xy_m)
              or delta[2] < -self.config.workspace_below_m
              or delta[2] > self.config.workspace_above_m):
            self.reason = "workspace"
        elif not self.release_intended and (
            np.linalg.norm(self._hand_offset() - self.initial_hand_offset) > self.config.drop_distance_m
            or self._no_grasp_s + 1e-12 >= self.config.grasp_loss_s
        ):
            self.reason = "dropped_peg"
        elif self.dwell_s + 1e-12 >= self.config.dwell_s:
            self.reason = "success"
        elif now - self.start_time + 1e-12 >= self.config.timeout_s:
            self.reason = "timeout"
        if self.reason is not None:
            self._finish_interval(now)
        return self.reason
