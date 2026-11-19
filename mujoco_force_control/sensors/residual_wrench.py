"""Prepare a simulated wrist wrench for the planned residual policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sensors.noisy_wrench import NoisyWrench


class WrenchSource(Protocol):
    def read_wrench(self) -> np.ndarray:
        """Return force XYZ in N and torque XYZ in N m, in the sensor frame."""


class WrenchInputError(RuntimeError):
    """A wrench input cannot safely be supplied to the simulation policy."""


@dataclass(frozen=True)
class WrenchSample:
    time_s: float
    wrench_local: np.ndarray


class ResidualWrenchInput:
    """Tare, perturb and delay one sensor sample per simulation physics step."""

    def __init__(self, source: WrenchSource, *, trial_seed: int, timestep_s: float) -> None:
        if type(trial_seed) is not int or trial_seed < 0:
            raise ValueError("trial_seed must be a non-negative integer")
        if not np.isfinite(timestep_s) or timestep_s <= 0:
            raise ValueError("timestep_s must be finite and positive")
        self.source = source
        self.timestep_s = float(timestep_s)
        self._fault: str | None = None
        self._last_update_time: float | None = None
        self._latest: WrenchSample | None = None
        self._tare = self._read_wrench()
        seed = int(np.random.SeedSequence([trial_seed, 2]).generate_state(1)[0])
        self._noise = NoisyWrench(
            seed=seed, noise_std=(0.05,) * 3 + (0.005,) * 3, latency_steps=2
        )

    @property
    def tare_local(self) -> np.ndarray:
        return self._tare.copy()

    def _fail(self, reason: str) -> None:
        self._fault = reason
        raise WrenchInputError(reason)

    def _check_fault(self) -> None:
        if self._fault is not None:
            raise WrenchInputError(f"wrench input faulted: {self._fault}")

    def _read_wrench(self) -> np.ndarray:
        value = np.asarray(self.source.read_wrench(), dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            self._fail("invalid wrist wrench: expected six finite values")
        return value.copy()

    def update(self, time_s: float) -> None:
        """Read at every physics step, including while the delay buffer fills."""
        self._check_fault()
        if not np.isfinite(time_s) or time_s < 0:
            self._fail("invalid wrench sample timestamp")
        if self._last_update_time is not None and not np.isclose(
            time_s - self._last_update_time, self.timestep_s, rtol=0, atol=1e-9
        ):
            self._fail("wrench samples must be collected once per physics step")
        wrench = self._read_wrench() - self._tare
        delayed = self._noise.update(float(time_s), wrench)
        self._last_update_time = float(time_s)
        if delayed is not None:
            self._latest = WrenchSample(delayed[0], delayed[1].copy())

    def sample(self, time_s: float) -> WrenchSample:
        """Return a fresh delayed sample, or raise without substituting a value."""
        self._check_fault()
        if not np.isfinite(time_s) or time_s < 0:
            self._fail("invalid control timestamp")
        if self._latest is None or self._last_update_time is None:
            self._fail("wrist wrench unavailable")
        if time_s < self._last_update_time:
            self._fail("wrist wrench is future-dated")
        age = float(time_s - self._latest.time_s)
        if age < 0:
            self._fail("wrist wrench is future-dated")
        if age > 0.1:
            self._fail("wrist wrench is stale")
        return WrenchSample(self._latest.time_s, self._latest.wrench_local.copy())
