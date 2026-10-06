"""PD position controller for joint-space control."""
import numpy as np


class PDController:
    """Proportional-Derivative controller for joint space position control."""

    def __init__(self, kp=100.0, kd=20.0, tau_max=87.0):
        """
        Args:
            kp: Proportional gain (7D)
            kd: Derivative gain (7D)
            tau_max: Maximum torque magnitude (saturate at this value)
        """
        self.kp = kp if isinstance(kp, np.ndarray) else np.full(7, kp)
        self.kd = kd if isinstance(kd, np.ndarray) else np.full(7, kd)
        self.tau_max = tau_max

    def compute(self, desired_pos, actual_pos, actual_vel, desired_vel=None):
        """Compute joint torques to reach desired position.

        Args:
            desired_pos: Desired joint positions (7D)
            actual_pos: Current joint positions (7D)
            actual_vel: Current joint velocities (7D)
            desired_vel: Desired joint velocities (7D, default: zero)

        Returns:
            Joint torques (7D)
        """
        if desired_vel is None:
            desired_vel = np.zeros(7)

        # Position error
        pos_error = desired_pos - actual_pos

        # Velocity error
        vel_error = desired_vel - actual_vel

        # PD law: tau = kp * e_pos + kd * e_vel
        tau = self.kp * pos_error + self.kd * vel_error

        # Saturate torques
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        return tau

    def set_gains(self, kp=None, kd=None):
        """Update controller gains."""
        if kp is not None:
            self.kp = kp if isinstance(kp, np.ndarray) else np.full(7, kp)
        if kd is not None:
            self.kd = kd if isinstance(kd, np.ndarray) else np.full(7, kd)
