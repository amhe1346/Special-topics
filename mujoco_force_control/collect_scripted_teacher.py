"""Collect timestamp-aligned privileged-oracle labels in the MuJoCo task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from envs.peg_in_hole import MAX_DEPTH, MAX_LATERAL_ERROR, MIN_DEPTH, PegInHoleEnv
from integrations.pi05_droid import CHUNK_SHAPE
from integrations.replay_provenance import replay_provenance
from integrations.rollout import RolloutConfig, run_rollout
from integrations.scripted_teacher import ScriptedInsertionTeacher


def _action_chunks(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != CHUNK_SHAPE[1]:
        raise ValueError(f"actions must have shape (samples, {CHUNK_SHAPE[1]})")
    if not np.all(np.isfinite(values)) or values.shape[0] < CHUNK_SHAPE[0]:
        raise ValueError(f"at least {CHUNK_SHAPE[0]} finite actions are required")
    return np.stack([
        values[start : start + CHUNK_SHAPE[0]]
        for start in range(values.shape[0] - CHUNK_SHAPE[0] + 1)
    ])


def _set_fixture_tilt(env: PegInHoleEnv, seed: int, tilt_range_deg: float) -> None:
    random = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    tilt = random.uniform(-tilt_range_deg, tilt_range_deg, 2)
    angle = float(np.deg2rad(np.linalg.norm(tilt)))
    if angle:
        quat = np.empty(4)
        mujoco.mju_axisAngle2Quat(
            quat,
            np.array([tilt[0], tilt[1], 0.0]) / np.linalg.norm(tilt),
            angle,
        )
        env.data.mocap_quat[env.fixture_mocap_id] = quat
        mujoco.mj_forward(env.model, env.data)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def collect(
    output: Path,
    seeds: dict[str, list[int]],
    *,
    lateral_range_m: float = 0.00005,
    tilt_range_deg: float = 0.03,
) -> None:
    if set(seeds) != {"train", "validation", "test"}:
        raise ValueError("train, validation and test splits are required")
    all_seeds = [seed for group in seeds.values() for seed in group]
    if (any(not isinstance(seed, int) or seed < 0 for seed in all_seeds)
            or any(not group for group in seeds.values())
            or len(set(all_seeds)) != len(all_seeds)):
        raise ValueError("splits must have nonempty, disjoint, non-negative integer seeds")
    if (not np.isfinite(lateral_range_m) or lateral_range_m < 0
            or not np.isfinite(tilt_range_deg) or not 0 <= tilt_range_deg < 90):
        raise ValueError("fixture randomization ranges must be finite and non-negative")
    output.mkdir(parents=True, exist_ok=False)

    teacher_config = {
        "frequency_hz": 15.0,
        "max_joint_speed_rad_s": 0.1,
        "cartesian_speed_m_s": 0.005,
        "target_depth_m": 0.014,
    }
    rollout_config = RolloutConfig(
        timeout_s=12.0,
        max_joint_speed_rad_s=teacher_config["max_joint_speed_rad_s"],
        frequency_hz=teacher_config["frequency_hz"],
        open_loop_horizon=1,
        no_contact=False,
    )
    manifest = {
        "schema": "stage-c-oracle-v2",
        "status": "in_progress",
        "label_source": "ScriptedInsertionTeacher",
        "label_kind": "15_step_executed_droid_normalized_joint_velocity_chunk",
        "action_horizon": CHUNK_SHAPE[0],
        "action_shape": list(CHUNK_SHAPE),
        "privileged_inputs": ["peg_tip_world_pose", "hole_entrance_pose", "fixture_pose"],
        "splits": seeds,
        "randomization": {
            "lateral_offset_range_m": [-lateral_range_m, lateral_range_m],
            "fixture_tilt_xy_range_deg": [-tilt_range_deg, tilt_range_deg],
        },
        "teacher_config": teacher_config,
        "rollout_config": asdict(rollout_config),
        "sample_count": 0,
        "trials": [],
        "replay_provenance": replay_provenance(),
        "limitations": [
            "Synthetic simulator-oracle labels, not human demonstrations.",
            "Oracle inputs are unavailable to pi0.5 at inference.",
            "Action chunks stack 15 consecutive executed 15 Hz labels from each successful trial.",
            "The normalized joint-velocity labels use DROID's 0.2-radian-per-command mapping; their speed range is narrower than DROID's full action range.",
            "The files need conversion to OpenPI's DROID/LeRobot dataset layout before fine-tuning.",
        ],
    }
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)

    try:
        for split, split_seeds in seeds.items():
            for seed in split_seeds:
                env = PegInHoleEnv()
                env.reset(seed=seed, lateral_range=lateral_range_m)
                _set_fixture_tilt(env, seed, tilt_range_deg)
                observations = []
                teacher = ScriptedInsertionTeacher(env, **teacher_config)
                with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                    result = run_rollout(
                        env,
                        renderer,
                        teacher,
                        rollout_config,
                        on_observation=observations.append,
                    )

                geometry = env.measure_geometry()
                if (
                    result["reason"] != "success"
                    or not MIN_DEPTH <= geometry.depth <= MAX_DEPTH
                    or geometry.lateral_error > MAX_LATERAL_ERROR
                    or geometry.angular_error > np.deg2rad(0.2)
                    or result["peak_force_n"] > 5.0
                ):
                    raise RuntimeError(
                        f"teacher trial failed validation for {split} seed {seed}: "
                        f"reason={result['reason']}, depth={geometry.depth:.6g}, "
                        f"lateral={geometry.lateral_error:.6g}, "
                        f"angle={geometry.angular_error:.6g}, "
                        f"peak_force={result['peak_force_n']:.6g}"
                    )
                if len(observations) != len(result["decisions"]):
                    raise RuntimeError("observation and teacher action counts do not match")
                if not observations:
                    raise RuntimeError(f"teacher trial produced no samples for seed {seed}")

                decisions = result["decisions"]
                executed = [decision["executed_actions"] for decision in decisions]
                if any(len(actions) != 1 for actions in executed):
                    raise RuntimeError("oracle collection requires one executed action per inference")
                teacher_actions = np.asarray([
                    decision["normalized_velocity"] + [float(decision["gripper_closed"])]
                    for decision in decisions
                ], dtype=np.float32)
                actions = np.asarray([
                    action_list[0]["normalized_velocity"]
                    + [float(action_list[0]["gripper_closed"])]
                    for action_list in executed
                ], dtype=np.float32)
                joint_targets = np.asarray([
                    action_list[0]["joint_target_rad"] for action_list in executed
                ], dtype=np.float32)
                sample_times = np.asarray(
                    [observation.time_s for observation in observations], dtype=np.float64
                )
                if (
                    not np.all(np.isfinite(actions))
                    or np.any(np.abs(actions[:, :7]) > 1.0)
                    or np.any(np.diff(sample_times) <= 0)
                ):
                    raise RuntimeError("oracle labels or sample timestamps are invalid")
                label_error = np.max(np.abs(actions[:, :7] - teacher_actions[:, :7]))
                if label_error > 1e-6:
                    raise RuntimeError(
                        f"teacher output differs from executed action by {label_error:.6g}"
                    )
                chunks = _action_chunks(actions)
                chunk_start_times = sample_times[: len(chunks)]

                filename = f"{split}-{seed}.npz"
                np.savez_compressed(
                    output / filename,
                    time_s=sample_times,
                    exterior_rgb=np.asarray(
                        [observation.exterior_rgb for observation in observations],
                        dtype=np.uint8,
                    ),
                    wrist_rgb=np.asarray(
                        [observation.wrist_rgb for observation in observations],
                        dtype=np.uint8,
                    ),
                    joint_position_rad=np.asarray(
                        [observation.joint_position_rad for observation in observations],
                        dtype=np.float32,
                    ),
                    gripper_closed_fraction=np.asarray(
                        [observation.gripper_closed_fraction for observation in observations],
                        dtype=np.float32,
                    ),
                    prompt=np.asarray([observation.prompt for observation in observations]),
                    teacher_action=teacher_actions,
                    action=actions,
                    action_chunk=chunks,
                    action_chunk_start_time_s=chunk_start_times,
                    action_chunk_start_index=np.arange(len(chunks), dtype=np.int32),
                    joint_target_rad=joint_targets,
                )
                manifest["trials"].append({
                    "split": split,
                    "seed": seed,
                    "file": filename,
                    "sample_count": len(observations),
                    "action_chunk_count": len(chunks),
                    "outcome": result["reason"],
                    "final_depth_m": geometry.depth,
                    "final_lateral_m": geometry.lateral_error,
                    "final_angle_rad": geometry.angular_error,
                    "peak_force_n": result["peak_force_n"],
                    "max_teacher_vs_executed_action_error": float(label_error),
                })
                manifest["sample_count"] += len(observations)
                _write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_manifest(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(100, 105)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(200, 205)))
    parser.add_argument("--lateral-range-mm", type=float, default=0.05)
    parser.add_argument("--tilt-range-deg", type=float, default=0.03)
    args = parser.parse_args()
    if args.lateral_range_mm < 0:
        parser.error("--lateral-range-mm must be non-negative")
    collect(
        args.output,
        {
            "train": args.train_seeds,
            "validation": args.validation_seeds,
            "test": args.test_seeds,
        },
        lateral_range_m=args.lateral_range_mm / 1000,
        tilt_range_deg=args.tilt_range_deg,
    )
    print(json.dumps({"output": str(args.output), "status": "complete"}, allow_nan=False))


if __name__ == "__main__":
    main()
