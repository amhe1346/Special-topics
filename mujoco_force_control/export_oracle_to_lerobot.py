"""Export the simulator oracle collection as split LeRobot DROID datasets.

Run this with the pinned OpenPI environment, which supplies LeRobot:

    .openpi/.venv/bin/python export_oracle_to_lerobot.py \
        results/stage_c_oracle_v2 --output results/stage_c_oracle_lerobot
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from integrations.pi05_droid import (
    CHUNK_SHAPE,
    DROID_MAX_JOINT_DELTA_RAD,
    droid_joint_velocity_to_delta,
)


FPS = 15
SIMULATION_TIMESTEP_S = 0.002
IMAGE_SHAPE = (224, 224, 3)
SPLIT_NAMES = ("train", "validation", "test")
FEATURES = {
    "exterior_image_1_left": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "exterior_image_2_left": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "wrist_image_left": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_position"],
    },
    "gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_position"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["actions"],
    },
    "simulation_time_s": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["simulation_time_s"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_timing(time_s: np.ndarray, path: Path) -> None:
    intervals = np.diff(time_s)
    if not np.all(intervals > 0):
        raise ValueError(f"{path}: simulation timestamps are not strictly increasing")
    expected_interval = 1.0 / FPS
    if (
        np.max(np.abs(intervals - expected_interval)) > SIMULATION_TIMESTEP_S + 1e-6
        or not np.isclose(
            np.mean(intervals),
            expected_interval,
            atol=SIMULATION_TIMESTEP_S / intervals.size + 1e-7,
            rtol=0,
        )
    ):
        raise ValueError(f"{path}: samples do not follow the 15 Hz action cadence")


def _read_trial(path: Path, manifest_trial: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    count = int(manifest_trial["sample_count"])
    action_chunk_count = int(manifest_trial["action_chunk_count"])
    expected_shapes = {
        "time_s": (count,),
        "exterior_rgb": (count, *IMAGE_SHAPE),
        "wrist_rgb": (count, *IMAGE_SHAPE),
        "joint_position_rad": (count, 7),
        "gripper_closed_fraction": (count,),
        "prompt": (count,),
        "action": (count, 8),
        "teacher_action": (count, 8),
        "action_chunk": (action_chunk_count, CHUNK_SHAPE[0], CHUNK_SHAPE[1]),
        "action_chunk_start_time_s": (action_chunk_count,),
        "action_chunk_start_index": (action_chunk_count,),
        "joint_target_rad": (count, 7),
    }
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} must have shape {shape}")
    if arrays["exterior_rgb"].dtype != np.uint8 or arrays["wrist_rgb"].dtype != np.uint8:
        raise ValueError(f"{path}: camera images must be uint8 RGB")
    for key in (
        "time_s",
        "joint_position_rad",
        "gripper_closed_fraction",
        "action",
        "teacher_action",
        "action_chunk",
        "action_chunk_start_time_s",
        "joint_target_rad",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"{path}: {key} contains non-finite values")

    if count < CHUNK_SHAPE[0]:
        raise ValueError(f"{path}: too few samples for the configured action horizon")
    _validate_timing(arrays["time_s"], path)
    expected_chunk_count = count - CHUNK_SHAPE[0] + 1
    if action_chunk_count != expected_chunk_count:
        raise ValueError(f"{path}: action chunk count is inconsistent with the source actions")
    chunk_starts = arrays["action_chunk_start_index"]
    if not np.array_equal(chunk_starts, np.arange(action_chunk_count)):
        raise ValueError(f"{path}: action chunks are not aligned to consecutive source steps")
    if not np.allclose(
        arrays["action_chunk_start_time_s"], arrays["time_s"][chunk_starts], atol=1e-7, rtol=0
    ):
        raise ValueError(f"{path}: action chunk timestamps do not match their source steps")
    expected_chunks = np.stack([
        arrays["action"][index : index + CHUNK_SHAPE[0]]
        for index in range(action_chunk_count)
    ])
    if not np.allclose(arrays["action_chunk"], expected_chunks, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: action chunks do not match the per-step action labels")
    prompts = arrays["prompt"].astype(str)
    if not prompts[0].strip() or np.any(prompts != prompts[0]):
        raise ValueError(f"{path}: task prompts must be nonempty and constant within a trial")

    actions = arrays["action"]
    teacher_actions = arrays["teacher_action"]
    if np.any(np.abs(actions[:, :7]) > 1.0) or np.any(np.abs(teacher_actions[:, :7]) > 1.0):
        raise ValueError(f"{path}: DROID joint-velocity actions must be in [-1, 1]")
    if np.any((arrays["gripper_closed_fraction"] < 0) | (arrays["gripper_closed_fraction"] > 1)):
        raise ValueError(f"{path}: gripper position state must be in [0, 1]")
    if not np.all(np.isin(actions[:, 7], (0.0, 1.0))):
        raise ValueError(f"{path}: DROID gripper-position targets must be binary 0/1")
    if not np.allclose(actions, teacher_actions, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: teacher labels differ from the executed action")

    # DROID's Franka joint-velocity adapter applies delta = action * 0.2 rad.
    actual_delta = arrays["joint_target_rad"] - arrays["joint_position_rad"]
    droid_delta = droid_joint_velocity_to_delta(actions[:, :7])
    max_error = float(np.max(np.abs(actual_delta - droid_delta)))
    if max_error > 2e-5:
        raise ValueError(
            f"{path}: normalized action does not reconstruct the executed "
            f"joint target under DROID scaling (max error {max_error:.6g} rad)"
        )
    if manifest_trial["outcome"] != "success":
        raise ValueError(f"{path}: only successful oracle episodes are exportable")
    return arrays


def _export_split(
    source_dir: Path,
    split_root: Path,
    split: str,
    trials: list[dict],
) -> dict:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed; run this script with "
            "mujoco_force_control/.openpi/.venv/bin/python"
        ) from error

    repo_id = f"special_topics/peg_in_hole_oracle_{split}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=split_root,
        robot_type="panda",
        fps=FPS,
        features=FEATURES,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    sample_count = 0
    trial_hashes = []
    for trial in trials:
        trial_path = source_dir / trial["file"]
        arrays = _read_trial(trial_path, trial)
        prompt = str(arrays["prompt"][0])
        for index in range(int(trial["sample_count"])):
            dataset.add_frame({
                "exterior_image_1_left": arrays["exterior_rgb"][index],
                # The policy input transform uses camera 1 and the wrist view.
                # Keep camera 2 present because the DROID LeRobot config requires it.
                "exterior_image_2_left": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
                "wrist_image_left": arrays["wrist_rgb"][index],
                "joint_position": arrays["joint_position_rad"][index].astype(np.float32),
                "gripper_position": np.asarray(
                    [arrays["gripper_closed_fraction"][index]], dtype=np.float32
                ),
                "actions": arrays["action"][index].astype(np.float32),
                "simulation_time_s": np.asarray([arrays["time_s"][index]], dtype=np.float32),
                "task": prompt,
            })
        dataset.save_episode()
        sample_count += int(trial["sample_count"])
        trial_hashes.append({
            "seed": trial["seed"],
            "file": trial["file"],
            "sha256": _sha256(trial_path),
            "samples": int(trial["sample_count"]),
        })
    return {
        "repo_id": repo_id,
        "path": str(split_root),
        "trial_count": len(trials),
        "sample_count": sample_count,
        "trials": trial_hashes,
    }


def export(source_dir: Path, output_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest not found: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("schema") != "stage-c-oracle-v2":
        raise ValueError("expected a complete stage-c-oracle-v2 source dataset")
    if source_manifest.get("status") != "complete":
        raise ValueError("source oracle dataset is not complete")
    if source_manifest.get("action_shape") != list(CHUNK_SHAPE):
        raise ValueError("source action chunk shape does not match the pinned policy")
    if source_dir in output_dir.parents or output_dir == source_dir:
        raise ValueError("output must not be the source directory or one of its parents")

    trial_groups = {
        split: [trial for trial in source_manifest["trials"] if trial["split"] == split]
        for split in SPLIT_NAMES
    }
    if any(not trials for trials in trial_groups.values()):
        raise ValueError("source dataset must contain nonempty train/validation/test splits")
    all_seeds = [trial["seed"] for trials in trial_groups.values() for trial in trials]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("source split seeds overlap")
    output_dir.mkdir(parents=True, exist_ok=False)

    conversion_manifest = {
        "schema": "stage-c-oracle-lerobot-v1",
        "status": "in_progress",
        "source": str(source_dir),
        "source_manifest_sha256": _sha256(manifest_path),
        "fps": FPS,
        "action_feature": "7 normalized DROID joint velocities + 1 normalized gripper position (0=open, 1=closed)",
        "joint_delta_scale_rad": DROID_MAX_JOINT_DELTA_RAD,
        "action_horizon": 16,
        "split_datasets": {},
        "limitations": [
            "Synthetic MuJoCo oracle demonstrations, not physical robot demonstrations.",
            "Normalized joint action uses the pinned DROID 0.2-radian-per-command mapping and the local rollout speed limit; gripper position uses DROID convention 0=open, 1=closed.",
            "The second external-camera feature is a black placeholder; the pinned DroidInputs transform consumes external camera 1 and the wrist camera.",
            "OpenPI fine-tuning must reuse the pi05_droid checkpoint DROID normalization assets.",
            "Action horizon 16 is set by the pinned OpenPI custom fine-tuning config; it is distinct from the inference checkpoint's 15-step horizon.",
        ],
    }
    report_path = output_dir / "conversion_manifest.json"
    report_path.write_text(json.dumps(conversion_manifest, indent=2, allow_nan=False) + "\n")
    try:
        conversion_manifest["lerobot_version"] = importlib.metadata.version("lerobot")
        for split in SPLIT_NAMES:
            conversion_manifest["split_datasets"][split] = _export_split(
                source_dir,
                output_dir / split,
                split,
                trial_groups[split],
            )
            report_path.write_text(json.dumps(conversion_manifest, indent=2, allow_nan=False) + "\n")
        conversion_manifest["status"] = "complete"
        report_path.write_text(json.dumps(conversion_manifest, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        conversion_manifest["status"] = "error"
        conversion_manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        report_path.write_text(json.dumps(conversion_manifest, indent=2, allow_nan=False) + "\n")
        raise
    return conversion_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="stage-c-oracle-v2 dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    report = export(args.source, args.output)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "splits": {
            split: {
                "repo_id": values["repo_id"],
                "trials": values["trial_count"],
                "samples": values["sample_count"],
            }
            for split, values in report["split_datasets"].items()
        },
    }, allow_nan=False))


if __name__ == "__main__":
    main()
