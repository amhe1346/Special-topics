"""Explicit, preview-only interface to the OpenPI pi05_droid checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
CHUNK_SHAPE = (15, 8)
DROID_MAX_JOINT_DELTA_RAD = 0.2


class Policy(Protocol):
    def infer(self, observation: dict) -> dict:
        """Return a pi05_droid action chunk."""


def _image_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("images must have shape (height, width, 3)")
    if value.dtype != np.uint8:
        raise ValueError("images must be uint8 RGB, not floats or BGR")
    height, width = value.shape[:2]
    ratio = max(width / IMAGE_SIZE, height / IMAGE_SIZE)
    new_width, new_height = int(width / ratio), int(height / ratio)
    if new_width < 1 or new_height < 1:
        raise ValueError("image aspect ratio is too extreme")
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(value).resize((new_width, new_height), resampling.BILINEAR)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    top = (IMAGE_SIZE - new_height) // 2
    left = (IMAGE_SIZE - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = np.asarray(resized)
    return canvas


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result.copy()


@dataclass(frozen=True)
class DroidObservation:
    """RGB images and state captured at one simulation timestamp."""

    time_s: float
    exterior_rgb: np.ndarray
    wrist_rgb: np.ndarray
    joint_position_rad: np.ndarray
    gripper_closed_fraction: float
    prompt: str

    def request(self) -> dict:
        """Build raw server input; OpenPI applies its own normalization."""
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("time_s must be finite and non-negative")
        if not np.isfinite(self.gripper_closed_fraction) or not 0 <= self.gripper_closed_fraction <= 1:
            raise ValueError("gripper_closed_fraction must be between 0 and 1")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty")
        return {
            "observation/exterior_image_1_left": _image_rgb(self.exterior_rgb),
            "observation/wrist_image_left": _image_rgb(self.wrist_rgb),
            "observation/joint_position": _vector(self.joint_position_rad, 7, "joint_position_rad"),
            "observation/gripper_position": np.array([self.gripper_closed_fraction], dtype=np.float64),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class ActionPreview:
    """Validated DROID action and suggested delta; never executes controls."""

    normalized_velocity: np.ndarray
    joint_delta_rad: np.ndarray
    gripper_closed: bool


def droid_joint_velocity_to_delta(normalized_velocity: np.ndarray) -> np.ndarray:
    """Convert DROID normalized joint velocities to one 15 Hz joint delta."""
    velocity = np.asarray(normalized_velocity, dtype=np.float64)
    if velocity.ndim < 1 or velocity.shape[-1] != 7:
        raise ValueError("normalized joint velocities must have a final dimension of 7")
    if not np.all(np.isfinite(velocity)) or np.any(np.abs(velocity) > 1.0):
        raise ValueError("normalized joint velocities must be finite and within [-1, 1]")
    return velocity * DROID_MAX_JOINT_DELTA_RAD


def preview_actions(response: dict) -> tuple[ActionPreview, ...]:
    """Validate all 15 actions; apply the example's clipping and gripper rule."""
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    actions = np.asarray(response["actions"], dtype=np.float64)
    if actions.shape != CHUNK_SHAPE or not np.all(np.isfinite(actions)):
        raise ValueError("policy actions must be a finite (15, 8) array")
    actions = np.clip(actions, -1.0, 1.0)
    return tuple(
        ActionPreview(
            row[:7].copy(),
            droid_joint_velocity_to_delta(row[:7]),
            bool(row[7] > 0.5),
        )
        for row in actions
    )


def infer_preview(policy: Policy, observation: DroidObservation) -> tuple[ActionPreview, ...]:
    """Call a supplied policy and validate its response without actuating."""
    return preview_actions(policy.infer(observation.request()))
