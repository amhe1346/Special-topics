"""Capture synchronized OpenPI-compatible observations from the task scene."""

from __future__ import annotations

import mujoco
import numpy as np

from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import DroidObservation


def capture_observation(
    env: PegInHoleEnv,
    renderer: mujoco.Renderer,
    *,
    prompt: str,
    exterior_camera: str = "task_left_camera",
) -> DroidObservation:
    """Capture two RGB views without advancing the simulation."""
    if exterior_camera not in ("task_left_camera", "task_right_camera"):
        raise ValueError("exterior_camera must be task_left_camera or task_right_camera")
    if not np.isfinite(env.data.time) or not np.all(np.isfinite(env.data.qpos)):
        raise ValueError("simulation state must be finite")
    renderer.update_scene(env.data, camera=exterior_camera)
    exterior = renderer.render().copy()
    renderer.update_scene(env.data, camera="task_wrist_camera")
    wrist = renderer.render().copy()
    joints = np.array([env.read_sensor(f"joint{i}_pos")[0] for i in range(1, 8)])
    finger_positions = env.data.qpos[env.model.jnt_qposadr[env.finger_joint_ids]]
    width = float(np.sum(finger_positions))
    if not -1e-5 <= width <= 0.08 + 1e-5:
        raise ValueError("simulated gripper width is outside its expected range")
    closed_fraction = float(np.clip(1.0 - width / 0.08, 0.0, 1.0))
    return DroidObservation(float(env.data.time), exterior, wrist, joints, closed_fraction, prompt)
