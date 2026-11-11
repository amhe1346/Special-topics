"""Seeded discrete-time wrench noise and fixed sensor latency."""

from collections import deque

import numpy as np


class NoisyWrench:
    """Delay noisy samples by an integer number of physics steps."""

    def __init__(self, *, seed: int, noise_std: tuple[float, ...], latency_steps: int) -> None:
        self.std = np.asarray(noise_std, dtype=float)
        if self.std.shape != (6,) or not np.all(np.isfinite(self.std)) or np.any(self.std < 0):
            raise ValueError("noise_std must contain six finite non-negative values")
        if not isinstance(latency_steps, int) or latency_steps < 0:
            raise ValueError("latency_steps must be a non-negative integer")
        self.rng = np.random.default_rng(seed)
        self.latency_steps = latency_steps
        self.samples: deque[tuple[float, np.ndarray]] = deque()
        self.last_time: float | None = None

    def update(self, time_s: float, wrench: np.ndarray) -> tuple[float, np.ndarray] | None:
        """Return the delayed timestamp and observation, or None while filling."""
        value = np.asarray(wrench, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)) or not np.isfinite(time_s):
            raise ValueError("wrench and timestamp must be finite")
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError("sample timestamps must strictly increase")
        self.last_time = time_s
        self.samples.append((time_s, value + self.rng.normal(0.0, self.std)))
        return self.samples.popleft() if len(self.samples) > self.latency_steps else None
