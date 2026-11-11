"""Cartesian force/torque feedback control for the Panda end effector."""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from sensors.load_cell import EndEffectorLoadCell


class EndEffectorForceController:
    """Apply proportional wrench feedback through the end-effector Jacobian.

    Target and measured wrenches are expressed in the load-cell frame. The
    requested correction is transformed to world coordinates before mapping it
    to joint torques with the Jacobian transpose.
    """

    def __init__(
        self,
        joint_controller: PDPositionController,
        load_cell: EndEffectorLoadCell | None = None,
        kp: float | tuple[float, ...] = (0.5, 0.5, 0.5, 0.2, 0.2, 0.2),
        gravity_compensation: bool = True,
        site_name: str = "ee_load_cell_site",
    ) -> None:
        self.joint_controller = joint_controller
        self.env = joint_controller.env
        self.load_cell = load_cell or EndEffectorLoadCell(self.env)
        self.gravity_compensation = gravity_compensation
        self.kp = self._gain_array(kp)
        self.site_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0:
            raise KeyError(f"Site {site_name!r} is not defined in the model")
        self._position_jacobian = np.zeros((3, self.env.model.nv))
        self._rotation_jacobian = np.zeros((3, self.env.model.nv))

    @staticmethod
    def _gain_array(kp: float | tuple[float, ...]) -> np.ndarray:
        gains = np.asarray(kp, dtype=float)
        try:
            gains = np.broadcast_to(gains, (6,)).copy()
        except ValueError as error:
            raise ValueError("kp must be scalar or contain six gains") from error
        if not np.all(np.isfinite(gains)) or np.any(gains < 0):
            raise ValueError("kp gains must be finite and non-negative")
        return gains

    def read_wrench(self) -> np.ndarray:
        """Return measured force and moment in the load-cell frame."""
        return self.load_cell.read_wrench()

    def wrench_to_joint_torques(self, wrench_world: np.ndarray) -> np.ndarray:
        """Map a world-frame wrench to joint-space torques."""
        wrench = np.asarray(wrench_world, dtype=float)
        if wrench.shape != (6,):
            raise ValueError("wrench_world must have shape (6,)")
        if not np.all(np.isfinite(wrench)):
            raise ValueError("wrench_world must contain only finite values")
        mujoco.mj_jacSite(
            self.env.model,
            self.env.data,
            self._position_jacobian,
            self._rotation_jacobian,
            self.site_id,
        )
        dofs = self.joint_controller.dof_indices
        return (
            self._position_jacobian[:, dofs].T @ wrench[:3]
            + self._rotation_jacobian[:, dofs].T @ wrench[3:]
        )

    def apply_wrench(self, wrench_world: np.ndarray) -> np.ndarray:
        """Apply a world-frame wrench using Jacobian-transpose torque mapping."""
        joint_torques = self.wrench_to_joint_torques(wrench_world)
        if self.gravity_compensation:
            joint_torques += self.env.data.qfrc_bias[
                self.joint_controller.dof_indices
            ]
        return self.joint_controller.apply_joint_torques(joint_torques)

    def step(self, target_wrench: np.ndarray) -> np.ndarray:
        """Apply one saturated wrench-feedback torque command.

        The target wrench has shape ``(6,)`` and is ordered as force (N)
        followed by moment (N m), both in the load-cell frame.
        """
        target = np.asarray(target_wrench, dtype=float)
        if target.shape != (6,):
            raise ValueError("target_wrench must have shape (6,)")
        if not np.all(np.isfinite(target)):
            raise ValueError("target_wrench must contain only finite values")

        wrench_correction = self.kp * (target - self.read_wrench())
        rotation = self.env.data.site_xmat[self.site_id].reshape(3, 3)
        wrench_world = np.concatenate(
            (rotation @ wrench_correction[:3], rotation @ wrench_correction[3:])
        )
        return self.apply_wrench(wrench_world)
