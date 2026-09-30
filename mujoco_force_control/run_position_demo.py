"""Run and plot a step or sinusoidal Panda joint-position trajectory."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from controllers.pd_position_controller import (
    ARM_JOINTS,
    DEFAULT_KD,
    DEFAULT_KP,
    PDPositionController,
)
from envs.base_env import BaseEnv


def run_demo(
    trajectory: str,
    duration: float,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
) -> tuple[np.ndarray, ...]:
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env, kp=kp, kd=kd)
    initial_position, _ = controller.read_joint_state()

    timestep = env.model.opt.timestep
    sample_count = int(round(duration / timestep))
    times = np.arange(sample_count) * timestep
    targets = np.tile(initial_position, (sample_count, 1))
    if trajectory == "step":
        targets[times >= 0.5, 0] += 0.2
    else:
        targets[:, 0] += 0.1 * np.sin(2 * np.pi * 0.5 * times)

    measured = np.empty_like(targets)
    for sample_index, target in enumerate(targets):
        controller.step(target)
        env.step()
        measured[sample_index], _ = controller.read_joint_state()

    return times, targets[:, 0], measured[:, 0], targets[:, 0] - measured[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=("step", "sine"), default="step")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    times, targets, measured, errors = run_demo(
        args.trajectory, args.duration, args.kp, args.kd
    )
    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(times, targets, label="target")
    axes[0].plot(times, measured, label="encoder")
    axes[0].set_ylabel(f"{ARM_JOINTS[0]} position (rad)")
    axes[0].legend()
    axes[1].plot(times, errors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("position error (rad)")
    axes[1].set_xlabel("time (s)")
    figure.suptitle(
        f"Panda PD {args.trajectory} response (kp={args.kp:g}, kd={args.kd:g})"
    )
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()