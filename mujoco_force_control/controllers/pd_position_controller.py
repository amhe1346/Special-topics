"""Encoder-feedback PD control for the Panda arm's direct-drive joints."""

from __future__ import annotations

import mujoco
import numpy as np

from envs.base_env import BaseEnv


ARM_JOINTS = tuple(f"joint{joint_number}" for joint_number in range(1, 8))
DEFAULT_KP = 400.0
DEFAULT_KD = 45.0


class PDPositionController:
    """Apply saturated PD torque commands from named joint position/velocity sensors."""

    def __init__(
        self,
        env: BaseEnv,
        joint_names: tuple[str, ...] = ARM_JOINTS,
        kp: float | tuple[float, ...] = DEFAULT_KP,
        kd: float | tuple[float, ...] = DEFAULT_KD,
        gravity_compensation: bool = True,
    ) -> None:
        self.env = env
        self.joint_names = tuple(joint_names)
        if not self.joint_names:
            raise ValueError("joint_names must contain at least one joint")

        self.kp = self._gain_array(kp, "kp")
        self.kd = self._gain_array(kd, "kd")
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("PD gains must be non-negative")

        self.joint_ids = np.array(
            [self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        )
        self.dof_indices = env.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array(
            [self._actuator_for_joint(joint_id, name) for joint_id, name in zip(self.joint_ids, self.joint_names)]
        )
        self.actuator_gears = env.model.actuator_gear[self.actuator_ids, 0]
        if np.any(np.isclose(self.actuator_gears, 0.0)):
            raise ValueError("joint actuators must have a non-zero transmission gear")
        self.gravity_compensation = gravity_compensation

        for name in self.joint_names:
            env.read_sensor(f"{name}_pos")
            env.read_sensor(f"{name}_vel")

    def _gain_array(self, gain: float | tuple[float, ...], name: str) -> np.ndarray:
        values = np.asarray(gain, dtype=float)
        try:
            return np.broadcast_to(values, (len(self.joint_names),)).copy()
        except ValueError as error:
            raise ValueError(f"{name} must be scalar or match joint_names") from error

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.env.model, object_type, name)
        if object_id < 0:
            raise KeyError(f"Joint {name!r} is not defined in the model")
        return object_id

    def _actuator_for_joint(self, joint_id: int, joint_name: str) -> int:
        matches = np.flatnonzero(
            (self.env.model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            & (self.env.model.actuator_trnid[:, 0] == joint_id)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one joint actuator for {joint_name!r}, found {len(matches)}"
            )
        return int(matches[0])

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read joint positions and velocities exclusively from encoder sensors."""
        positions = np.array(
            [self.env.read_sensor(f"{name}_pos")[0] for name in self.joint_names]
        )
        velocities = np.array(
            [self.env.read_sensor(f"{name}_vel")[0] for name in self.joint_names]
        )
        return positions, velocities

    def step(self, target_position: np.ndarray) -> np.ndarray:
        """Compute and apply one saturated PD torque command."""
        target = np.asarray(target_position, dtype=float)
        if target.shape != (len(self.joint_names),):
            raise ValueError(f"target_position must have shape ({len(self.joint_names)},)")
        if not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain only finite values")

        positions, velocities = self.read_joint_state()
        joint_torques = self.kp * (target - positions) - self.kd * velocities
        if self.gravity_compensation:
            joint_torques += self.env.data.qfrc_bias[self.dof_indices]

        controls = joint_torques / self.actuator_gears
        for control_index, actuator_id in enumerate(self.actuator_ids):
            if self.env.model.actuator_ctrllimited[actuator_id]:
                control_range = self.env.model.actuator_ctrlrange[actuator_id]
                controls[control_index] = np.clip(
                    controls[control_index], control_range[0], control_range[1]
                )
            if self.env.model.actuator_forcelimited[actuator_id]:
                force_range = self.env.model.actuator_forcerange[actuator_id]
                controls[control_index] = np.clip(
                    controls[control_index], force_range[0], force_range[1]
                )

        self.env.data.ctrl[self.actuator_ids] = controls
        return controls * self.actuator_gears