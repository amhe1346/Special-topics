"""End-effector force/torque sensor interface."""

from __future__ import annotations

import numpy as np

from envs.base_env import BaseEnv


class EndEffectorLoadCell:
    """Read the simulated six-axis load cell in its sensor-local frame."""

    def __init__(
        self,
        env: BaseEnv,
        force_sensor: str = "ee_force",
        torque_sensor: str = "ee_torque",
    ) -> None:
        self.env = env
        self.force_sensor = force_sensor
        self.torque_sensor = torque_sensor
        self.read_force()
        self.read_torque()

    def read_force(self) -> np.ndarray:
        """Return the measured force vector in newtons."""
        force = self.env.read_sensor(self.force_sensor)
        if force.shape != (3,):
            raise ValueError(f"Force sensor {self.force_sensor!r} must have 3 values")
        return force

    def read_torque(self) -> np.ndarray:
        """Return the measured moment vector in newton-metres."""
        torque = self.env.read_sensor(self.torque_sensor)
        if torque.shape != (3,):
            raise ValueError(f"Torque sensor {self.torque_sensor!r} must have 3 values")
        return torque

    def read_wrench(self) -> np.ndarray:
        """Return force and moment concatenated in sensor-local coordinates."""
        return np.concatenate((self.read_force(), self.read_torque()))
