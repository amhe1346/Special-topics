"""Evaluate one-step reconstruction of recorded motion under the rollout adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from envs.peg_contact import PegContactForce
from envs.peg_in_hole import PegInHoleEnv
from integrations.replay_provenance import replay_provenance
from integrations.rollout import RolloutConfig


def motion_action(start: np.ndarray, end: np.ndarray, elapsed_s: float,
                  frequency_hz: float = 15.0) -> np.ndarray:
    """Convert achieved displacement to a hypothetical unclipped DROID command.

    This is an outcome-derived label, not a reconstruction of expert intent.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (7,) or end.shape != (7,) or not np.all(np.isfinite([start, end])):
        raise ValueError("joint positions must be finite seven-element vectors")
    if not np.isfinite(elapsed_s) or elapsed_s <= 0 or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("elapsed time and frequency must be finite and positive")
    return (end - start) / (elapsed_s * frequency_hz * 0.2)


def evaluate_trial(samples_path: Path, steps_path: Path, config: RolloutConfig,
                   refine_iterations: int = 0) -> dict:
    """Reset to each recorded state and compare a single adapter-controlled interval."""
    if not isinstance(refine_iterations, int) or refine_iterations < 0:
        raise ValueError("refine_iterations must be a non-negative integer")
    env = PegInHoleEnv()
    controller = PDPositionController(env)
    bounds = env.model.jnt_range[controller.joint_ids]
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    expected_size = mujoco.mj_stateSize(env.model, state_spec)
    with np.load(samples_path, allow_pickle=False) as samples, np.load(steps_path, allow_pickle=False) as steps:
        times = samples["time_s"]
        indices = samples["step_index"]
        positions = samples["joint_position_rad"]
        states = steps["integration_state"]
        step_times = steps["time_s"]
        controls = steps["executed_ctrl"]
        if (len(times) < 2 or len(indices) != len(times) or positions.shape != (len(times), 7)
                or states.ndim != 2 or states.shape[1] != expected_size
                or controls.shape != (len(states), env.model.nu)
                or not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls))
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions))
                or not np.allclose(np.diff(step_times), env.model.opt.timestep, atol=1e-9, rtol=0)
                or np.any(np.diff(times) <= 0) or np.any(np.diff(indices) <= 0)
                or indices[0] < 0 or indices[-1] >= len(states)
                or not np.allclose(times, step_times[indices], atol=1e-9, rtol=0)):
            raise ValueError("sample and physics-step timestamps or shapes are invalid")
        joint_errors = []
        tip_errors = []
        peaks = []
        clipped = []
        reference_joint_errors = []
        reference_tip_errors = []
        for i in range(len(times) - 1):
            elapsed = float(times[i + 1] - times[i])
            action = motion_action(positions[i], positions[i + 1], elapsed, config.frequency_hz)
            clipped_action = np.clip(action, -1, 1)
            delta = np.clip(clipped_action * 0.2, -config.max_joint_speed_rad_s / config.frequency_hz,
                            config.max_joint_speed_rad_s / config.frequency_hz)
            mujoco.mj_setState(env.model, env.data, states[indices[i]], state_spec)
            mujoco.mj_forward(env.model, env.data)
            for step_index in range(int(indices[i]) + 1, int(indices[i + 1]) + 1):
                env.data.ctrl[:] = controls[step_index]
                env.step()
                mujoco.mj_forward(env.model, env.data)
            reference_joints, _ = controller.read_joint_state()
            reference_joint_errors.append(float(np.max(np.abs(reference_joints - positions[i + 1]))))
            reference_tip_errors.append(float(np.linalg.norm(env.data.site_xpos[env.peg_tip_id]
                                                             - samples["peg_tip_world_m"][i + 1])))
            force = PegContactForce(env)
            was_clipped = bool(np.any(np.abs(action * 0.2 - delta) > 1e-12))
            for attempt in range(refine_iterations + 1):
                mujoco.mj_setState(env.model, env.data, states[indices[i]], state_spec)
                mujoco.mj_forward(env.model, env.data)
                target = np.clip(positions[i] + delta, bounds[:, 0], bounds[:, 1])
                was_clipped |= bool(np.any(np.abs(target - (positions[i] + delta)) > 1e-12))
                peak = 0.0
                for _ in range(int(indices[i + 1] - indices[i])):
                    controller.step(target)
                    env.step()
                    mujoco.mj_forward(env.model, env.data)
                    peak = max(peak, force.measure().magnitude_sum)
                actual_joints, _ = controller.read_joint_state()
                if attempt < refine_iterations:
                    proposed = delta + positions[i + 1] - actual_joints
                    delta = np.clip(proposed, -config.max_joint_speed_rad_s / config.frequency_hz,
                                    config.max_joint_speed_rad_s / config.frequency_hz)
                    was_clipped |= bool(np.any(np.abs(proposed - delta) > 1e-12))
            clipped.append(was_clipped)
            joint_errors.append(float(np.max(np.abs(actual_joints - positions[i + 1]))))
            tip_errors.append(float(np.linalg.norm(env.data.site_xpos[env.peg_tip_id]
                                                   - samples["peg_tip_world_m"][i + 1])))
            peaks.append(peak)
    return {
        "intervals": len(joint_errors),
        "max_joint_error_rad": max(joint_errors),
        "p95_joint_error_rad": float(np.percentile(joint_errors, 95)),
        "max_tip_error_m": max(tip_errors),
        "p95_tip_error_m": float(np.percentile(tip_errors, 95)),
        "clipped_intervals": sum(clipped),
        "peak_replay_contact_n": max(peaks),
        "replay_intervals_over_5n": sum(peak > 5.0 for peak in peaks),
        "reference_replay_max_joint_error_rad": max(reference_joint_errors),
        "reference_replay_max_tip_error_m": max(reference_tip_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", type=Path, help="new JSON report file; never overwritten")
    parser.add_argument("--refine-iterations", type=int, default=0,
                        help="try bounded label refinement using local simulation replay")
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest.get("schema") != "stage-c-development-v2" or manifest.get("status") != "complete":
        raise ValueError("expected a complete stage-c-development-v2 dataset")
    if manifest.get("replay_provenance") != replay_provenance():
        raise ValueError("dataset simulator provenance does not match the current environment")
    config = RolloutConfig(no_contact=False)
    results = [{"seed": trial["seed"], **evaluate_trial(args.dataset / trial["file"],
                args.dataset / trial["steps_file"], config, args.refine_iterations)}
               for trial in manifest["trials"] if trial["split"] == args.split]
    if not results:
        raise ValueError("selected split has no trials")
    report = json.dumps({"dataset": str(args.dataset), "split": args.split,
                         "replay_provenance": manifest["replay_provenance"],
                         "rollout_config": asdict(config), "termination_rules_applied": False,
                         "refine_iterations": args.refine_iterations,
                         "results": results}, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as output:
            output.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
