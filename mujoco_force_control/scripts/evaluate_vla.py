#!/usr/bin/env python3
"""Evaluate π₀.₅ policy against PD baseline in MuJoCo."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

from mujoco_force_control.envs.base_env import ForceControlEnv
from mujoco_force_control.controllers.pd_position_controller import PDController
from mujoco_force_control.vla.pi05_wrapper import Pi05Wrapper
from mujoco_force_control.data.language_sampler import LanguageSampler


def compute_metrics(joint_pos_trajectory, joint_targets_trajectory):
    """Compute tracking metrics.

    Args:
        joint_pos_trajectory: (T, 7) actual joint positions
        joint_targets_trajectory: (T, 7) target joint positions

    Returns:
        Dict with metrics
    """
    pos_error = joint_pos_trajectory - joint_targets_trajectory
    rmse = np.sqrt(np.mean(pos_error**2))
    max_error = np.max(np.abs(pos_error))

    # Settling time (first time error < 10% of initial, stays there)
    initial_error = np.abs(pos_error[0])
    threshold = initial_error * 0.1
    settled_idx = np.where(np.all(np.abs(pos_error[i:]) < threshold, axis=0) for i in range(len(pos_error)))[0]
    settling_time = settled_idx[0] if len(settled_idx) > 0 else float('inf')

    return {
        'rmse': rmse,
        'max_error': max_error,
        'mean_error': np.mean(np.abs(pos_error)),
        'settling_time': settling_time,
    }


def evaluate_policy(policy_name, env, policy_fn, num_trials=5, trial_duration=500):
    """Evaluate a control policy.

    Args:
        policy_name: Name of policy (for logging)
        env: ForceControlEnv
        policy_fn: Function that takes obs and returns action
        num_trials: Number of trials
        trial_duration: Steps per trial

    Returns:
        Dict with aggregated metrics
    """
    all_metrics = []
    all_trajectories = {
        'actual': [],
        'target': [],
    }

    for trial in range(num_trials):
        obs = env.reset()
        joint_pos_traj = [obs['joint_pos']]
        joint_target_traj = []

        # Random target (step to home)
        target = np.array([0, -0.5, 0, -1.5, 0, 1.5, 0.5])

        for step in range(trial_duration):
            action = policy_fn(obs, target)
            joint_target_traj.append(action)

            obs, _, _, _ = env.step(action)
            joint_pos_traj.append(obs['joint_pos'])

        joint_pos_traj = np.array(joint_pos_traj[:-1])  # Remove last extra step
        joint_target_traj = np.array(joint_target_traj)

        # Compute metrics
        metrics = compute_metrics(joint_pos_traj, joint_target_traj)
        all_metrics.append(metrics)

        all_trajectories['actual'].append(joint_pos_traj)
        all_trajectories['target'].append(joint_target_traj)

    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)

    return aggregated, all_trajectories


def evaluate_baselines(env, num_trials=5):
    """Evaluate PD baseline.

    Args:
        env: ForceControlEnv
        num_trials: Number of trials

    Returns:
        Dict with baseline metrics
    """
    controller = PDController(kp=100.0, kd=20.0)

    def pd_policy(obs, target):
        return controller.compute(target, obs['joint_pos'], obs['joint_vel'])

    metrics, trajectories = evaluate_policy(
        "PD Baseline",
        env,
        pd_policy,
        num_trials=num_trials
    )

    return metrics, trajectories


def evaluate_pi05(env, model_path, num_trials=5):
    """Evaluate π₀.₅ policy.

    Args:
        env: ForceControlEnv
        model_path: Path to trained model checkpoint
        num_trials: Number of trials

    Returns:
        Dict with π₀.₅ metrics
    """
    if not Path(model_path).exists():
        print(f"Warning: Model not found at {model_path}")
        return None, None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi05Wrapper(model_path=model_path, device=device)

    def pi05_policy(obs, target):
        # Prepare inputs for π₀.₅
        try:
            image = env.render_rgb(height=224, width=224)
        except:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        images = np.expand_dims(image, axis=(0, 1)).astype(np.float32) / 255.0  # (1, 1, 3, 224, 224)

        # Proprioceptive state
        proprio = np.concatenate([
            obs['joint_pos'],
            obs['joint_vel'],
            obs['ee_force'],
        ]).astype(np.float32)
        proprio = np.expand_dims(proprio, axis=(0, 1))  # (1, 1, 16)

        # Language embedding (dummy)
        language_emb = np.random.randn(1, 256).astype(np.float32)

        # Get action
        action = model.predict(images, proprio, language_emb)[0, 0]
        return action

    metrics, trajectories = evaluate_policy(
        "π₀.₅",
        env,
        pi05_policy,
        num_trials=num_trials
    )

    return metrics, trajectories


def main(model_path="checkpoints/pi05_best.pt", num_trials=5):
    """Run full benchmark.

    Args:
        model_path: Path to trained π₀.₅ model
        num_trials: Number of trials per policy
    """
    env = ForceControlEnv(robot="panda", render_mode=None)

    print("=" * 60)
    print("Evaluating π₀.₅ vs. PD Baseline")
    print("=" * 60)

    # Evaluate baseline
    print("\nEvaluating PD baseline...")
    pd_metrics, pd_traj = evaluate_baselines(env, num_trials=num_trials)

    print(f"PD Baseline Results:")
    print(f"  RMSE: {pd_metrics['rmse_mean']:.4f} ± {pd_metrics['rmse_std']:.4f} rad")
    print(f"  Max Error: {pd_metrics['max_error_mean']:.4f} ± {pd_metrics['max_error_std']:.4f} rad")
    print(f"  Settling Time: {pd_metrics['settling_time_mean']:.1f} ± {pd_metrics['settling_time_std']:.1f} steps")

    # Evaluate π₀.₅
    print(f"\nEvaluating π₀.₅ model...")
    pi05_metrics, pi05_traj = evaluate_pi05(env, model_path, num_trials=num_trials)

    if pi05_metrics is not None:
        print(f"\nπ₀.₅ Results:")
        print(f"  RMSE: {pi05_metrics['rmse_mean']:.4f} ± {pi05_metrics['rmse_std']:.4f} rad")
        print(f"  Max Error: {pi05_metrics['max_error_mean']:.4f} ± {pi05_metrics['max_error_std']:.4f} rad")
        print(f"  Settling Time: {pi05_metrics['settling_time_mean']:.1f} ± {pi05_metrics['settling_time_std']:.1f} steps")

        # Comparison
        print(f"\nComparison (π₀.₅ vs. PD):")
        rmse_ratio = pi05_metrics['rmse_mean'] / pd_metrics['rmse_mean']
        print(f"  RMSE ratio: {rmse_ratio:.2f}x")

        if rmse_ratio < 1.2:
            print(f"  ✓ π₀.₅ performs within 20% of baseline!")
        elif rmse_ratio < 1.5:
            print(f"  ~ π₀.₅ within 50% of baseline (reasonable for first training)")
        else:
            print(f"  ✗ π₀.₅ underperforming; more training needed")

        # Save results
        results = {
            'pd': pd_metrics,
            'pi05': pi05_metrics,
            'comparison': {
                'rmse_ratio': rmse_ratio,
            }
        }
        np.save('evaluation_results.npy', results, allow_pickle=True)
        print(f"\n✓ Results saved to evaluation_results.npy")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description="Evaluate π₀.₅ policy")
    parser.add_argument("--model_path", type=str, default="checkpoints/pi05_best.pt", help="Path to trained model")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of evaluation trials")

    args = parser.parse_args()

    main(model_path=args.model_path, num_trials=args.num_trials)
