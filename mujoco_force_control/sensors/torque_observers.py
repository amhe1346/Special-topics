"""Joint-space disturbance torque observers for the Panda arm."""

from __future__ import annotations

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController


class ExtendedKalmanTorqueObserver:
    """Estimate joint disturbance torques with an augmented-state EKF.

    The state contains joint positions, velocities, and disturbance torques.
    MuJoCo's mass matrix and bias/passive forces define the nonlinear state
    transition; encoder sensors provide position and velocity measurements.
    """

    def __init__(
        self,
        controller: PDPositionController,
        torque_process_noise: float = 1.0,
        position_measurement_noise: float = 1e-6,
        velocity_measurement_noise: float = 1e-4,
        initial_torque_variance: float = 100.0,
    ) -> None:
        self.controller = controller
        self.env = controller.env
        self.joint_count = len(controller.joint_names)
        self.torque_process_noise = self._positive(
            torque_process_noise, "torque_process_noise", allow_zero=True
        )
        self.position_measurement_noise = self._positive(
            position_measurement_noise, "position_measurement_noise"
        )
        self.velocity_measurement_noise = self._positive(
            velocity_measurement_noise, "velocity_measurement_noise"
        )
        self.initial_torque_variance = self._positive(
            initial_torque_variance, "initial_torque_variance"
        )
        self.state = np.zeros(3 * self.joint_count)
        self.covariance = np.diag(
            np.concatenate(
                (
                    np.full(self.joint_count, self.position_measurement_noise),
                    np.full(self.joint_count, self.velocity_measurement_noise),
                    np.full(self.joint_count, self.initial_torque_variance),
                )
            )
        )
        self.measurement_noise = np.diag(
            np.concatenate(
                (
                    np.full(self.joint_count, self.position_measurement_noise),
                    np.full(self.joint_count, self.velocity_measurement_noise),
                )
            )
        )
        self.process_noise = np.diag(
            np.concatenate(
                (
                    np.full(self.joint_count, 1e-10),
                    np.full(self.joint_count, 1e-8),
                    np.full(self.joint_count, self.torque_process_noise),
                )
            )
        )
        self.measurement_matrix = np.zeros((2 * self.joint_count, 3 * self.joint_count))
        indices = np.arange(self.joint_count)
        self.measurement_matrix[indices, indices] = 1.0
        self.measurement_matrix[
            self.joint_count + indices, self.joint_count + indices
        ] = 1.0
        self._data = mujoco.MjData(self.env.model)
        self._full_mass = np.empty((self.env.model.nv, self.env.model.nv))
        self._initialized = False

    @staticmethod
    def _positive(value: float, name: str, allow_zero: bool = False) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
        return value

    def _acceleration(self, state: np.ndarray, actuator_torques: np.ndarray) -> np.ndarray:
        count = self.joint_count
        positions = state[:count]
        velocities = state[count : 2 * count]
        disturbance_torques = state[2 * count :]

        self._data.qpos[:] = self.env.data.qpos
        self._data.qvel[:] = self.env.data.qvel
        self._data.qpos[self.controller.env.model.jnt_qposadr[self.controller.joint_ids]] = (
            positions
        )
        self._data.qvel[self.controller.dof_indices] = velocities
        mujoco.mj_forward(self.env.model, self._data)
        mujoco.mj_fullM(self.env.model, self._data, self._full_mass)
        dofs = self.controller.dof_indices
        mass_matrix = self._full_mass[np.ix_(dofs, dofs)]
        generalized_forces = (
            actuator_torques
            + disturbance_torques
            - self._data.qfrc_bias[dofs]
            + self._data.qfrc_passive[dofs]
        )
        return np.linalg.solve(mass_matrix, generalized_forces)

    def _transition(
        self, state: np.ndarray, actuator_torques: np.ndarray, dt: float
    ) -> np.ndarray:
        count = self.joint_count
        acceleration = self._acceleration(state, actuator_torques)
        next_state = state.copy()
        next_state[:count] += dt * state[count : 2 * count] + 0.5 * dt**2 * acceleration
        next_state[count : 2 * count] += dt * acceleration
        return next_state

    def _transition_jacobian(
        self, state: np.ndarray, actuator_torques: np.ndarray, dt: float
    ) -> np.ndarray:
        jacobian = np.eye(3 * self.joint_count)
        for index in range(3 * self.joint_count):
            step = 1e-5 * max(1.0, abs(state[index]))
            upper = state.copy()
            lower = state.copy()
            upper[index] += step
            lower[index] -= step
            jacobian[:, index] = (
                self._transition(upper, actuator_torques, dt)
                - self._transition(lower, actuator_torques, dt)
            ) / (2.0 * step)
        return jacobian

    def update(
        self,
        actuator_torques: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Update from encoder measurements and return estimated disturbance torques."""
        interval = self.env.model.opt.timestep if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("dt must be finite and positive")
        torques = (
            self.controller.read_joint_torques()
            if actuator_torques is None
            else np.asarray(actuator_torques, dtype=float)
        )
        if torques.shape != (self.joint_count,) or not np.all(np.isfinite(torques)):
            raise ValueError(
                f"actuator_torques must contain {self.joint_count} finite values"
            )

        positions, velocities = self.controller.read_joint_state()
        measurement = np.concatenate((positions, velocities))
        count = self.joint_count
        if not self._initialized:
            self.state[:count] = positions
            self.state[count : 2 * count] = velocities
            self._initialized = True
            return self.state[2 * count :].copy()

        jacobian = self._transition_jacobian(self.state, torques, interval)
        predicted_state = self._transition(self.state, torques, interval)
        predicted_covariance = (
            jacobian @ self.covariance @ jacobian.T + interval * self.process_noise
        )
        innovation = measurement - self.measurement_matrix @ predicted_state
        innovation_covariance = (
            self.measurement_matrix
            @ predicted_covariance
            @ self.measurement_matrix.T
            + self.measurement_noise
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            self.measurement_matrix @ predicted_covariance,
        ).T
        self.state = predicted_state + kalman_gain @ innovation
        residual_projection = np.eye(3 * count) - kalman_gain @ self.measurement_matrix
        self.covariance = (
            residual_projection @ predicted_covariance @ residual_projection.T
            + kalman_gain @ self.measurement_noise @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self.state[2 * count :].copy()

    def reset(self) -> None:
        """Forget the current estimate and initialize again from the next measurement."""
        self.state.fill(0.0)
        self._initialized = False
        self.covariance = np.diag(
            np.concatenate(
                (
                    np.full(self.joint_count, self.position_measurement_noise),
                    np.full(self.joint_count, self.velocity_measurement_noise),
                    np.full(self.joint_count, self.initial_torque_variance),
                )
            )
        )


class SlidingModeTorqueObserver:
    """Estimate disturbance torque from inverse-dynamics residuals."""

    def __init__(
        self,
        controller: PDPositionController,
        sliding_gain: float = 50.0,
        boundary_layer: float = 0.1,
    ) -> None:
        self.controller = controller
        self.env = controller.env
        self.joint_count = len(controller.joint_names)
        self.sliding_gain = ExtendedKalmanTorqueObserver._positive(
            sliding_gain, "sliding_gain"
        )
        self.boundary_layer = ExtendedKalmanTorqueObserver._positive(
            boundary_layer, "boundary_layer"
        )
        self.estimate = np.zeros(self.joint_count)
        self._previous_velocities: np.ndarray | None = None
        self._full_mass = np.empty((self.env.model.nv, self.env.model.nv))

    def update(
        self,
        actuator_torques: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Update from joint-velocity changes and return estimated disturbance torques."""
        interval = self.env.model.opt.timestep if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("dt must be finite and positive")
        torques = (
            self.controller.read_joint_torques()
            if actuator_torques is None
            else np.asarray(actuator_torques, dtype=float)
        )
        if torques.shape != (self.joint_count,) or not np.all(np.isfinite(torques)):
            raise ValueError(
                f"actuator_torques must contain {self.joint_count} finite values"
            )

        _, velocities = self.controller.read_joint_state()
        if self._previous_velocities is None:
            self._previous_velocities = velocities
            return self.estimate.copy()

        accelerations = (velocities - self._previous_velocities) / interval
        self._previous_velocities = velocities
        mujoco.mj_forward(self.env.model, self.env.data)
        mujoco.mj_fullM(self.env.model, self.env.data, self._full_mass)
        dofs = self.controller.dof_indices
        mass_matrix = self._full_mass[np.ix_(dofs, dofs)]
        residual = (
            mass_matrix @ accelerations
            + self.env.data.qfrc_bias[dofs]
            - self.env.data.qfrc_passive[dofs]
            - torques
        )
        sliding_signal = np.clip(
            (residual - self.estimate) / self.boundary_layer, -1.0, 1.0
        )
        self.estimate += interval * self.sliding_gain * sliding_signal
        return self.estimate.copy()

    def reset(self) -> None:
        """Clear the estimate and velocity history."""
        self.estimate.fill(0.0)
        self._previous_velocities = None
