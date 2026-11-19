"""Exploratory simulation reward, independent of policy observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


FAILURES = frozenset({
    "dropped_peg", "force_limit", "workspace", "timeout", "release_requested",
})
INVALID = frozenset({"invalid_state", "wrench_input_error", "inference_timeout", "stale_observation"})


@dataclass(frozen=True)
class TaskState:
    lateral_error_m: float
    depth_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lateral_error_m) or self.lateral_error_m < 0:
            raise ValueError("lateral error must be finite and non-negative")
        if not math.isfinite(self.depth_m):
            raise ValueError("depth must be finite")


@dataclass(frozen=True)
class ForceSample:
    contact_force_n: float
    duration_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.contact_force_n) or self.contact_force_n < 0:
            raise ValueError("contact force must be finite and non-negative")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("sample duration must be finite and positive")


@dataclass(frozen=True)
class RewardBreakdown:
    progress: float
    terminal: float
    force_excess_n_s: float
    force_penalty: float
    total: float


def potential(state: TaskState) -> float:
    return 0.5 * math.exp(-state.lateral_error_m / 0.00008) + 0.5 * min(max(state.depth_m / 0.020, 0), 1)


def interval_reward(
    previous: TaskState,
    current: TaskState,
    force_samples: Sequence[ForceSample],
    *,
    force_aware: bool,
    termination: str | None = None,
) -> RewardBreakdown:
    """Score one interval; invalid or unknown terminal reasons cannot be scored."""
    if type(force_aware) is not bool:
        raise ValueError("force_aware must be a bool")
    if termination in INVALID:
        raise ValueError(f"invalid trial cannot be scored: {termination}")
    if termination is not None and termination != "success" and termination not in FAILURES:
        raise ValueError(f"unknown termination reason: {termination}")
    excess = math.fsum(
        max(sample.contact_force_n - 1.0, 0.0) * sample.duration_s for sample in force_samples
    )
    progress = potential(current) - potential(previous)
    terminal = 10.0 if termination == "success" else -5.0 if termination in FAILURES else 0.0
    penalty = -excess if force_aware else 0.0
    return RewardBreakdown(progress, terminal, excess, penalty, progress + terminal + penalty)
