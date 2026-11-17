"""Summarize training demonstrations and save example policy camera views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from export_oracle_to_lerobot import _read_trial


def inspect(source: Path, output: Path) -> dict:
    manifest_path = source / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "stage-c-oracle-v2" or manifest.get("status") != "complete":
        raise ValueError("expected a complete stage-c-oracle-v2 manifest")
    trials = [trial for trial in manifest["trials"] if trial["split"] == "train"]
    if not trials or {trial["seed"] for trial in trials} != set(manifest["splits"]["train"]):
        raise ValueError("training trials do not match training split")
    output.mkdir(parents=True, exist_ok=False)

    actions = []
    image_changes = {"exterior": [], "wrist": []}
    for trial in trials:
        filename = Path(trial["file"])
        if filename.name != str(filename):
            raise ValueError("trial filename must be a single path component")
        arrays = _read_trial(source / filename, trial)
        actions.append(arrays["action"])
        for name, key in (("exterior", "exterior_rgb"), ("wrist", "wrist_rgb")):
            images = arrays[key].astype(np.float32)
            image_changes[name].extend(
                np.mean(np.abs(np.diff(images, axis=0)), axis=(1, 2, 3)).tolist()
            )
        if trial is trials[0]:
            indices = [0, len(arrays["time_s"]) // 2, len(arrays["time_s"]) - 1]
            canvas = Image.new("RGB", (224 * 3, 224 * 2 + 44), "white")
            draw = ImageDraw.Draw(canvas)
            for col, index in enumerate(indices):
                for row, key in enumerate(("exterior_rgb", "wrist_rgb")):
                    canvas.paste(Image.fromarray(arrays[key][index]), (col * 224, row * 224 + 22))
                draw.text((col * 224 + 3, 3), f"frame {index}, t={arrays['time_s'][index]:.2f}s", fill="black")
            draw.text((3, 22 + 224 * 2 + 3), "Top: exterior camera   Bottom: wrist camera", fill="black")
            canvas.save(output / "training_camera_examples.png")

    action_values = np.concatenate(actions)
    velocities = action_values[:, :7]
    report = {
        "schema": "stage-c-demonstration-inspection-v1",
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split": "train",
        "trials": len(trials),
        "frames": len(action_values),
        "example_seed": trials[0]["seed"],
        "joint_action_abs_max": float(np.max(np.abs(velocities))),
        "joint_action_abs_p95": float(np.percentile(np.abs(velocities), 95)),
        "gripper_closed_fraction_of_labels": float(np.mean(action_values[:, 7] > 0.5)),
        "adjacent_frame_mean_absolute_rgb_change": {
            name: {
                "median_on_0_to_255_scale": float(np.median(values)),
                "p95_on_0_to_255_scale": float(np.percentile(values, 95)),
            }
            for name, values in image_changes.items()
        },
        "limitations": [
            "Pixel change is not a visibility or learnability test.",
            "The oracle uses simulator geometry and internal target history unavailable to the policy.",
            "Training frames are correlated within episodes; the split contains no policy recovery trajectories.",
            "The images show only examples, not proof that clearance-scale alignment is observable.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/stage_c_oracle_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
