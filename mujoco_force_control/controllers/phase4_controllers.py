"""Hybrid, cascaded-force, impedance, and admittance controllers."""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.force_controller import EndEffectorForceController
from controllers.pd_position_controller import PDPositionController


def _vector(
    value: float | tuple[float, ...] | np.ndarray,
    name: str,
    size: int,
    *,
    positive: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    try:
        values = np.broadcast_to(values, (size,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or contain {size} values") from error
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return values


def _rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew_vector = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    if np.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * skew_vector


class HybridPositionForceController:
    """Track joint positions while applying force feedback on selected wrench axes."""

    def __init__(
        self,
        joint_controller: PDPositionController,
        force_controller: EndEffectorForceController | None = None,
        force_axes: tuple[bool, ...] = (False, False, True, False, False, False),
    ) -> None:
        self.joint_controller = joint_controller
        self.force_controller = force_controller or EndEffectorForceController(
            joint_controller
        )
        self.force_axes = np.asarray(force_axes, dtype=bool)
        if self.force_axes.shape != (6,):
            raise ValueError("force_axes must contain six boolean values")

    def step(
        self, target_joint_positions: np.ndarray, target_wrench: np.ndarray
    ) -> np.ndarray:
        """Apply joint-position control plus selected Cartesian force feedback."""
        target = np.asarray(target_joint_positions, dtype=float)
        if target.shape != (len(self.joint_controller.joint_names),):
            raise ValueError(
                "target_joint_positions must match the configured joint count"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_joint_positions must contain only finite values")
        target_wrench = np.asarray(target_wrench, dtype=float)
        if target_wrench.shape != (6,) or not np.all(np.isfinite(target_wrench)):
            raise ValueError("target_wrench must contain six finite values")

        positions, velocities = self.joint_controller.read_joint_state()
        joint_torques = self.joint_controller.kp * (target - positions)
        joint_torques -= self.joint_controller.kd * velocities
        if self.joint_controller.gravity_compensation:
            joint_torques += self.joint_controller.env.data.qfrc_bias[
                self.joint_controller.dof_indices
            ]

        force_controller = self.force_controller
        wrench_correction = force_controller.kp * (
            target_wrench - force_controller.read_wrench()
        )
        wrench_correction[~self.force_axes] = 0.0
        rotation = force_controller.env.data.site_xmat[
            force_controller.site_id
        ].reshape(3, 3)
        wrench_world = np.concatenate(
            (
                rotation @ wrench_correction[:3],
                rotation @ wrench_correction[3:],
            )
        )
        joint_torques += force_controller.wrench_to_joint_torques(wrench_world)
        return self.joint_controller.apply_joint_torques(joint_torques)


class CascadedForceController:
    """Use an outer PI force loop to set the target for an inner wrench loop."""

    def __init__(
        self,
        force_controller: EndEffectorForceController,
        kp: float | tuple[float, ...] = 0.2,
        ki: float | tuple[float, ...] = 1.0,
        integral_limit: float | tuple[float, ...] = 20.0,
        wrench_limit: float | tuple[float, ...] = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0),
    ) -> None:
        self.force_controller = force_controller
        self.kp = _vector(kp, "kp", 6)
        self.ki = _vector(ki, "ki", 6)
        self.integral_limit = _vector(integral_limit, "integral_limit", 6)
        self.wrench_limit = _vector(wrench_limit, "wrench_limit", 6, positive=True)
        self.integral_error = np.zeros(6)
        self.last_target_wrench = np.zeros(6)

    def step(self, target_wrench: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Update the outer PI loop and apply the inner wrench feedback command."""
        target = np.asarray(target_wrench, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("target_wrench must contain six finite values")
        interval = (
            self.force_controller.env.model.opt.timestep if dt is None else float(dt)
        )
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("dt must be finite and positive")

        error = target - self.force_controller.read_wrench()
        candidate_integral = np.clip(
            self.integral_error + interval * error,
            -self.integral_limit,
            self.integral_limit,
        )
        raw_wrench = target + self.kp * error + self.ki * candidate_integral
        limited_wrench = np.clip(raw_wrench, -self.wrench_limit, self.wrench_limit)
        saturated_outward = (raw_wrench != limited_wrench) & (
            error * (raw_wrench - limited_wrench) > 0.0
        )
        self.integral_error = np.where(
            saturated_outward, self.integral_error, candidate_integral
        )
        self.last_target_wrench = limited_wrench
        return self.force_controller.step(limited_wrench)

    def reset(self) -> None:
        """Clear the outer-loop integral state."""
        self.integral_error.fill(0.0)
        self.last_target_wrench.fill(0.0)


class CartesianImpedanceController:
    """Track a Cartesian pose with virtual spring-damper wrench feedback."""

    def __init__(
        self,
        force_controller: EndEffectorForceController,
        stiffness: float | tuple[float, ...] = (200.0, 200.0, 200.0, 20.0, 20.0, 20.0),
        damping: float | tuple[float, ...] = (30.0, 30.0, 30.0, 5.0, 5.0, 5.0),
    ) -> None:
        self.force_controller = force_controller
        self.env = force_controller.env
        self.stiffness = _vector(stiffness, "stiffness", 6)
        self.damping = _vector(damping, "damping", 6)
        self._position_jacobian = np.zeros((3, self.env.model.nv))
        self._rotation_jacobian = np.zeros((3, self.env.model.nv))

    def step(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None = None,
        feedforward_wrench: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply spring-damper wrench control toward a world-frame Cartesian pose."""
        position_target = np.asarray(target_position, dtype=float)
        if position_target.shape != (3,) or not np.all(np.isfinite(position_target)):
            raise ValueError("target_position must contain three finite values")
        site_id = self.force_controller.site_id
        current_position = self.env.data.site_xpos[site_id]
        current_rotation = self.env.data.site_xmat[site_id].reshape(3, 3)
        desired_rotation = (
            current_rotation
            if target_rotation is None
            else np.asarray(target_rotation, dtype=float)
        )
        if desired_rotation.shape != (3, 3) or not np.all(
            np.isfinite(desired_rotation)
        ):
            raise ValueError("target_rotation must be a finite 3-by-3 matrix")
        if not np.allclose(
            desired_rotation.T @ desired_rotation, np.eye(3), atol=1e-5
        ) or not np.isclose(np.linalg.det(desired_rotation), 1.0, atol=1e-5):
            raise ValueError("target_rotation must be a proper rotation matrix")

        mujoco.mj_jacSite(
            self.env.model,
            self.env.data,
            self._position_jacobian,
            self._rotation_jacobian,
            site_id,
        )
        velocity = np.concatenate(
            (
                self._position_jacobian @ self.env.data.qvel,
                self._rotation_jacobian @ self.env.data.qvel,
            )
        )
        pose_error = np.concatenate(
            (
                position_target - current_position,
                _rotation_error(current_rotation, desired_rotation),
            )
        )
        wrench = self.stiffness * pose_error - self.damping * velocity
        if feedforward_wrench is not None:
            feedforward = np.asarray(feedforward_wrench, dtype=float)
            if feedforward.shape != (6,) or not np.all(np.isfinite(feedforward)):
                raise ValueError("feedforward_wrench must contain six finite values")
            wrench += feedforward
        return self.force_controller.apply_wrench(wrench)


class CartesianAdmittanceController:
    """Adapt a Cartesian pose using virtual mass-damper-spring force dynamics."""

    def __init__(
        self,
        impedance_controller: CartesianImpedanceController,
        mass: float | tuple[float, ...] = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1),
        damping: float | tuple[float, ...] = (20.0, 20.0, 20.0, 2.0, 2.0, 2.0),
        stiffness: float | tuple[float, ...] = 0.0,
        max_displacement: float | tuple[float, ...] = (0.1, 0.1, 0.1, 0.5, 0.5, 0.5),
    ) -> None:
        self.impedance_controller = impedance_controller
        self.force_controller = impedance_controller.force_controller
        self.env = self.force_controller.env
        self.mass = _vector(mass, "mass", 6, positive=True)
        self.damping = _vector(damping, "damping", 6)
        self.stiffness = _vector(stiffness, "stiffness", 6)
        self.max_displacement = _vector(
            max_displacement, "max_displacement", 6, positive=True
        )
        self.displacement = np.zeros(6)
        self.velocity = np.zeros(6)
        self._nominal_position = self.env.data.site_xpos[
            self.force_controller.site_id
        ].copy()
        self._nominal_rotation = self.env.data.site_xmat[
            self.force_controller.site_id
        ].reshape(3, 3).copy()

    def step(
        self,
        target_wrench: np.ndarray,
        dt: float | None = None,
        target_position: np.ndarray | None = None,
        target_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate wrench error into a bounded pose offset and track that pose."""
        target = np.asarray(target_wrench, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("target_wrench must contain six finite values")
        interval = self.env.model.opt.timestep if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("dt must be finite and positive")
        if target_position is not None:
            self._nominal_position = np.asarray(target_position, dtype=float)
            if self._nominal_position.shape != (3,) or not np.all(
                np.isfinite(self._nominal_position)
            ):
                raise ValueError("target_position must contain three finite values")
        if target_rotation is not None:
            rotation = np.asarray(target_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("target_rotation must be a finite 3-by-3 matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                raise ValueError("target_rotation must be a proper rotation matrix")
            self._nominal_rotation = rotation

        error = self.force_controller.read_wrench() - target
        acceleration = (
            error - self.damping * self.velocity - self.stiffness * self.displacement
        ) / self.mass
        self.velocity += interval * acceleration
        self.displacement += interval * self.velocity
        bounded_displacement = np.clip(
            self.displacement, -self.max_displacement, self.max_displacement
        )
        clipped = bounded_displacement != self.displacement
        self.velocity[clipped] = 0.0
        self.displacement = bounded_displacement

        rotation = self._nominal_rotation
        position = self._nominal_position + rotation @ self.displacement[:3]
        desired_rotation = rotation @ _rotation_matrix(self.displacement[3:])
        return self.impedance_controller.step(position, desired_rotation)

    def reset(
        self, target_position: np.ndarray | None = None, target_rotation: np.ndarray | None = None
    ) -> None:
        """Clear virtual motion and optionally set a new nominal pose."""
        self.displacement.fill(0.0)
        self.velocity.fill(0.0)
        if target_position is None:
            self._nominal_position = self.env.data.site_xpos[
                self.force_controller.site_id
            ].copy()
        else:
            position = np.asarray(target_position, dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("target_position must contain three finite values")
            self._nominal_position = position
        if target_rotation is None:
            self._nominal_rotation = self.env.data.site_xmat[
                self.force_controller.site_id
            ].reshape(3, 3).copy()
        else:
            rotation = np.asarray(target_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("target_rotation must be a finite 3-by-3 matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                raise ValueError("target_rotation must be a proper rotation matrix")
            self._nominal_rotation = rotation
