#!/usr/bin/env python3
"""Collect training data by running the PD controller in MuJoCo."""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from mujoco_force_control.envs.base_env import ForceControlEnv
from mujoco_force_control.controllers.pd_position_controller import PDController
from mujoco_force_control.data.dataset import TrajectoryRecorder
from mujoco_force_control.data.language_sampler import LanguageSampler


def generate_target_trajectory(scenario, num_steps, dt=0.002):
    """Generate target joint positions over time.

    Args:
        scenario: One of ['sinusoid', 'step', 'random_walk']
        num_steps: Number of timesteps
        dt: Simulation timestep

    Returns:
        (num_steps, 7) array of joint position targets
    """
    targets = np.zeros((num_steps, 7))

    if scenario == 'sinusoid':
        # Oscillate each joint at different frequencies
        for i in range(7):
            freq = 0.5 + i * 0.1  # Hz
            targets[:, i] = 0.5 * np.sin(2 * np.pi * freq * np.arange(num_steps) * dt)

    elif scenario == 'step':
        # Step to a fixed target position (home)
        target_home = np.array([0, -0.5, 0, -1.5, 0, 1.5, 0.5])
        targets[:] = target_home

    else:  # random_walk
        # Random walk in joint space
        targets[0] = np.random.uniform(-1, 1, 7)
        for t in range(1, num_steps):
            targets[t] = targets[t-1] + np.random.randn(7) * 0.02
            targets[t] = np.clip(targets[t], -2.5, 2.5)

    return targets


def collect_rollout(env, controller, scenario, num_steps=500):
    """Collect one rollout with given scenario.

    Args:
        env: ForceControlEnv instance
        controller: PDController instance
        scenario: One of ['sinusoid', 'step', 'random_walk']
        num_steps: Duration of rollout in timesteps

    Returns:
        TrajectoryRecorder with collected data
    """
    obs = env.reset()
    recorder = TrajectoryRecorder()

    # Generate target trajectory
    targets = generate_target_trajectory(scenario, num_steps, dt=env.dt)

    for step in range(num_steps):
        # Render RGB observation
        try:
            image = env.render_rgb(height=224, width=224)
        except Exception:
            # If rendering fails, use a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Compute control action (PD law)
        target_pos = targets[step]
        action = controller.compute(
            desired_pos=target_pos,
            actual_pos=obs['joint_pos'],
            actual_vel=obs['joint_vel'],
        )

        # Record this step
        recorder.append(image, obs, target_pos, env.data.time)

        # Step environment
        obs, _, _, _ = env.step(action)

    return recorder


def collect_dataset(num_rollouts=300, rollout_steps=500, output_dir="dataset/rollouts", num_workers=1):
    """Collect a dataset of rollouts.

    Args:
        num_rollouts: Total number of rollouts to collect
        rollout_steps: Duration of each rollout
        output_dir: Directory to save HDF5 files
        num_workers: Number of parallel workers (currently serial, parallel via Ray in future)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment and controller
    env = ForceControlEnv(robot="panda", render_mode=None)
    controller = PDController(kp=100.0, kd=20.0)

    scenarios = ['sinusoid', 'step', 'random_walk']

    print(f"Collecting {num_rollouts} rollouts to {output_dir}/")
    print(f"Using scenarios: {scenarios}")

    for rollout_idx in tqdm(range(num_rollouts), desc="Collecting rollouts"):
        # Sample scenario
        scenario = np.random.choice(scenarios)

        # Collect rollout
        recorder = collect_rollout(env, controller, scenario, num_steps=rollout_steps)

        # Generate language label
        language = LanguageSampler.sample(scenario)

        # Save to HDF5
        output_path = output_dir / f"rollout_{rollout_idx:06d}.h5"
        metadata = {
            'scenario': scenario,
            'language': language,
            'duration_steps': recorder.length(),
            'dt': env.dt,
        }
        recorder.save_hdf5(output_path, metadata)

    print(f"✓ Collected {num_rollouts} rollouts")
    print(f"  Saved to: {output_dir}")
    print(f"  Total size: {sum(f.stat().st_size for f in output_dir.glob('*.h5')) / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data from PD controller")
    parser.add_argument("--num_rollouts", type=int, default=50, help="Number of rollouts to collect")
    parser.add_argument("--rollout_steps", type=int, default=500, help="Steps per rollout")
    parser.add_argument("--output_dir", type=str, default="dataset/rollouts", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")

    args = parser.parse_args()

    collect_dataset(
        num_rollouts=args.num_rollouts,
        rollout_steps=args.rollout_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
