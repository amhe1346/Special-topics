"""Oracle-guided simulation teacher for DROID-style action-pipeline checks."""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.phase4_controllers import _rotation_error
from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import CHUNK_SHAPE


TEACHER_IK_GAIN = 0.005
TEACHER_MAX_TARGET_STEP_RAD = 0.002


class ScriptedInsertionTeacher:
    """Generate bounded joint-delta actions using privileged MuJoCo geometry.

    This is a simulator oracle for data-pipeline experiments, not a human
    demonstration or a deployable policy.
    """

    def __init__(
        self,
        env: PegInHoleEnv,
        *,
        frequency_hz: float = 15.0,
        max_joint_speed_rad_s: float = 0.1,
        cartesian_speed_m_s: float = 0.005,
        target_depth_m: float = 0.014,
    ) -> None:
        values = (frequency_hz, max_joint_speed_rad_s, cartesian_speed_m_s, target_depth_m)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("teacher rates and speeds must be finite and positive")
        self.env = env
        self.period_s = 1.0 / frequency_hz
        self.cartesian_speed_m_s = cartesian_speed_m_s
        self.target_depth_m = target_depth_m
        self.max_delta_rad = max_joint_speed_rad_s * self.period_s
        self.action_delta_scale_rad = 0.2
        self.control_substeps = int(round(self.period_s / env.model.opt.timestep))
        if self.control_substeps < 1:
            raise ValueError("teacher frequency must not exceed the simulator frequency")

        self.joint_ids = np.array([
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ])
        if np.any(self.joint_ids < 0):
            raise ValueError("teacher requires the seven Panda arm joints")
        self.dof_indices = env.model.jnt_dofadr[self.joint_ids]
        self.site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_load_cell_site"
        )
        if self.site_id < 0:
            raise ValueError("teacher requires the ee_load_cell_site")

        self.jac_position = np.zeros((3, env.model.nv))
        self.jac_rotation = np.zeros((3, env.model.nv))
        self.target_joint: np.ndarray | None = None
        self.target_position: np.ndarray | None = None
        self.target_rotation: np.ndarray | None = None
        self.target_position_floor: float | None = None

    def _initialize_target(self) -> None:
        env = self.env
        data = env.data
        self.target_joint = np.array([
            env.read_sensor(f"joint{i}_pos")[0] for i in range(1, 8)
        ])
        self.target_position = data.site_xpos[self.site_id].copy()
        self.target_rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()
        fixture = data.mocap_pos[env.fixture_mocap_id]
        peg_tip = data.site_xpos[env.peg_tip_id]
        self.target_position[:2] += fixture[:2] - peg_tip[:2]
        self.target_position_floor = self.target_position[2] - 0.038

    def _action(self) -> np.ndarray:
        env = self.env
        data = env.data
        if self.target_joint is None:
            self._initialize_target()
        assert self.target_joint is not None
        assert self.target_position is not None
        assert self.target_rotation is not None
        assert self.target_position_floor is not None

        current_position = data.site_xpos[self.site_id].copy()
        current_rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()
        geometry = env.measure_geometry()
        if geometry.depth < self.target_depth_m:
            self.target_position[2] = max(
                self.target_position_floor,
                self.target_position[2] - self.cartesian_speed_m_s * self.period_s,
            )

        rotation_delta = _rotation_error(current_rotation, self.target_rotation)
        mujoco.mj_jacSite(
            env.model, data, self.jac_position, self.jac_rotation, self.site_id
        )
        jacobian = np.vstack((
            self.jac_position[:, self.dof_indices],
            self.jac_rotation[:, self.dof_indices],
        ))
        error = np.r_[self.target_position - current_position, rotation_delta]
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.0025 * np.eye(6), error
        )

        # Approximate the benchmark controller's repeated per-physics-step IK
        # updates over one policy interval.
        gain = 1.0 - (1.0 - TEACHER_IK_GAIN) ** self.control_substeps
        self.target_joint += np.clip(
            joint_delta * gain,
            -TEACHER_MAX_TARGET_STEP_RAD * self.control_substeps,
            TEACHER_MAX_TARGET_STEP_RAD * self.control_substeps,
        )
        current_joint = np.array([
            env.read_sensor(f"joint{i}_pos")[0] for i in range(1, 8)
        ])
        normalized_delta = np.clip(
            self.target_joint - current_joint,
            -self.max_delta_rad,
            self.max_delta_rad,
        ) / self.action_delta_scale_rad
        return np.r_[normalized_delta, 1.0]

    def infer(self, observation: dict) -> dict:
        """Return a constant chunk; the rollout adapter executes its first row."""
        if not isinstance(observation, dict):
            raise ValueError("observation must be a policy-input dictionary")
        action = self._action()
        chunk = np.repeat(action[np.newaxis, :], CHUNK_SHAPE[0], axis=0)
        return {"actions": chunk}
