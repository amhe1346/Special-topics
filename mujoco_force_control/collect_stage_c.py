"""Collect small, simulation-only classical benchmark development trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from benchmark_insertion import TrialConfig, run_trial
from envs.peg_contact import PegContactForce
from integrations.sim_observation import capture_observation
from sensors.load_cell import EndEffectorLoadCell


class Recorder:
    """Capture post-step state and the controls that drove the preceding step."""

    def __init__(self, period_s: float = 0.1) -> None:
        self.period_s = period_s
        self.next_time: float | None = None
        self.renderer: mujoco.Renderer | None = None
        self.load_cell: EndEffectorLoadCell | None = None
        self.contact: PegContactForce | None = None
        self.rows: list[dict] = []

    def __call__(self, env) -> None:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.model, height=224, width=224)
            self.load_cell = EndEffectorLoadCell(env)
            self.contact = PegContactForce(env)
            self.next_time = float(env.data.time)
        if env.data.time + 1e-10 < self.next_time:
            return
        observation = capture_observation(env, self.renderer, prompt="Insert the peg into the hole.")
        geometry = env.measure_geometry()
        assert self.load_cell is not None and self.contact is not None
        force = self.contact.measure()
        self.rows.append({
            "time_s": observation.time_s,
            "exterior_rgb": observation.exterior_rgb,
            "wrist_rgb": observation.wrist_rgb,
            "joint_position_rad": observation.joint_position_rad,
            "joint_velocity_rad_s": np.array([env.read_sensor(f"joint{i}_vel")[0] for i in range(1, 8)]),
            "gripper_closed_fraction": observation.gripper_closed_fraction,
            "wrench_local": self.load_cell.read_wrench(),
            "contact_force_n": force.magnitude_sum,
            "contact_force_world_n": force.world_vector,
            "executed_ctrl": env.data.ctrl.copy(),
            "peg_tip_world_m": env.data.site_xpos[env.peg_tip_id].copy(),
            "hole_entrance_world_m": env.data.site_xpos[env.hole_site_id].copy(),
            "depth_m": geometry.depth,
            "lateral_m": geometry.lateral_error,
            "angle_rad": geometry.angular_error,
            "phase": 0 if geometry.depth < 0 else 1,
        })
        self.next_time += self.period_s

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def collect(output: Path, seeds: dict[str, list[int]], method: str = "impedance") -> None:
    """Write trial-separated samples and fit normalization on training trials only."""
    if set(seeds) != {"train", "validation", "test"}:
        raise ValueError("train, validation and test splits are required")
    all_seeds = [seed for group in seeds.values() for seed in group]
    if any(not isinstance(seed, int) or seed < 0 for seed in all_seeds):
        raise ValueError("seeds must be non-negative integers")
    if any(not group for group in seeds.values()) or len(set(all_seeds)) != len(all_seeds):
        raise ValueError("splits must be nonempty and have disjoint seeds")
    output.mkdir(parents=True, exist_ok=False)
    training = []
    manifest = {"schema": "stage-c-development-v1", "status": "in_progress",
                "method": method, "splits": seeds, "trials": []}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for split, group in seeds.items():
        for seed in group:
            recorder = Recorder()
            try:
                result = run_trial(method, TrialConfig(seed=seed), on_step=recorder)
            finally:
                recorder.close()
            if not recorder.rows:
                raise RuntimeError(f"No recorded samples for seed {seed}")
            arrays = {key: np.asarray([row[key] for row in recorder.rows]) for key in recorder.rows[0]}
            trial_file = f"{split}-{seed}.npz"
            np.savez_compressed(output / trial_file, **arrays)
            manifest["trials"].append({"split": split, "seed": seed, "file": trial_file, "outcome": result})
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            if split == "train":
                training.append(arrays["joint_position_rad"])
    values = np.concatenate(training)
    normalization = {"field": "joint_position_rad", "mean": np.mean(values, axis=0).tolist(),
                     "std": np.std(values, axis=0).tolist(), "training_sample_count": len(values),
                     "training_seeds": seeds["train"]}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2, allow_nan=False) + "\n")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--method", choices=("position", "hybrid", "impedance"), default="impedance")
    args = parser.parse_args()
    collect(args.output, {"train": [0, 1, 2], "validation": [10], "test": [11]}, args.method)


if __name__ == "__main__":
    main()
