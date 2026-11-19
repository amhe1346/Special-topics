"""Record wrist and peg-contact forces during a classical insertion trial."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from benchmark_insertion import TrialConfig, run_trial
from controllers.pd_position_controller import PDPositionController
from envs.peg_contact import PegContactForce
from integrations.pi05_droid import preview_actions
from integrations.replay_provenance import replay_provenance
from integrations.residual_adapter import ForceResidualObservation
from integrations.residual_experiments import Variant, get_config
from sensors.load_cell import EndEffectorLoadCell
from sensors.residual_wrench import ResidualWrenchInput


class ShadowResidual:
    """Observe adapter inputs without changing the classical controller."""

    def __init__(self) -> None:
        self.calls = 0
        self.force_calls = 0

    def infer(self, observation):
        self.calls += 1
        self.force_calls += isinstance(observation, ForceResidualObservation)
        return np.zeros(7)


def run_probe(variant: Variant, seed: int) -> tuple[dict, list[dict]]:
    config = TrialConfig(seed=seed)
    experiment = get_config(variant)
    policy = ShadowResidual()
    sensor = None
    load_cell = None
    contact = None
    controller = None
    adapter = None
    start = None
    tick = 0
    rows = []
    nominal = preview_actions({"actions": np.tile(np.r_[np.zeros(7), 1.0], (15, 1))})[0]

    def record(env):
        nonlocal sensor, load_cell, contact, controller, adapter, start, tick
        time_s = float(env.data.time)
        if sensor is None:
            load_cell = EndEffectorLoadCell(env)
            sensor = ResidualWrenchInput(load_cell, trial_seed=seed, timestep_s=env.model.opt.timestep)
            contact = PegContactForce(env)
            controller = PDPositionController(env)
        assert load_cell is not None and contact is not None and controller is not None
        raw = load_cell.read_wrench()
        sensor.update(time_s)
        sample = None
        if len(rows) >= 2:
            sample = sensor.sample(time_s)
            if adapter is None:
                start = time_s
                adapter = experiment.make_adapter(
                    policy, sensor, start_time_s=start, physics_timestep_s=env.model.opt.timestep,
                )
            assert start is not None
            expected = start + tick / 15
            if abs(time_s - expected) <= env.model.opt.timestep / 2 + 1e-9:
                joints, velocity = controller.read_joint_state()
                fingers = env.data.qpos[env.model.jnt_qposadr[env.finger_joint_ids]]
                adapter.step(
                    time_s=time_s, joint_position_rad=joints, joint_velocity_rad_s=velocity,
                    gripper_closed_fraction=float(np.clip(1 - np.sum(fingers) / 0.08, 0, 1)),
                    nominal=nominal, joint_bounds_rad=env.model.jnt_range[controller.joint_ids],
                )
                tick += 1
        force = contact.measure()
        row = {
            "time_s": time_s,
            "contact_force_n": float(force.magnitude_sum),
            "wrench_sample_time_s": None if sample is None else sample.time_s,
        }
        for i, name in enumerate(("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")):
            row[f"raw_{name}"] = float(raw[i])
            row[f"tared_{name}"] = float(raw[i] - sensor.tare_local[i])
            row[f"observed_{name}"] = None if sample is None else float(sample.wrench_local[i])
        rows.append(row)

    outcome = run_trial("position", config, on_step=record)
    if not rows:
        raise RuntimeError("contact trial produced no samples")
    valid = [row for row in rows if row["wrench_sample_time_s"] is not None]
    if not valid:
        raise RuntimeError("contact trial produced no delayed wrench samples")
    contact_rows = [row for row in rows if row["contact_force_n"] > 0.1]
    precontact = [row for row in rows if row["contact_force_n"] <= 0.1]
    if not contact_rows or not precontact:
        raise RuntimeError("trial did not contain both contact and no-contact samples")
    delays = [row["time_s"] - row["wrench_sample_time_s"] for row in valid]

    def max_tared_force(samples):
        return float(max(np.linalg.norm([row["tared_fx_n"], row["tared_fy_n"], row["tared_fz_n"]])
                         for row in samples))

    report = {
        "schema": "stage-d-shadow-contact-probe-v1",
        "status": "complete",
        "experiment": asdict(experiment),
        "trial_config": asdict(config),
        "outcome": outcome,
        "replay_provenance": replay_provenance(),
        "samples": len(rows),
        "timestep_s": outcome["timestep_s"],
        "contact_threshold_n": 0.1,
        "contact_samples_above_threshold": len(contact_rows),
        "peak_contact_force_n": float(max(row["contact_force_n"] for row in rows)),
        "max_tared_wrist_force_no_contact_n": max_tared_force(precontact),
        "max_tared_wrist_force_contact_n": max_tared_force(contact_rows),
        "first_contact_time_s": contact_rows[0]["time_s"],
        "sample_delay_min_s": min(delays),
        "sample_delay_max_s": max(delays),
        "shadow_policy_calls": policy.calls,
        "shadow_force_observation_calls": policy.force_calls,
        "limitations": [
            "Classical position controller drives the trial; the zero-action adapter runs in shadow.",
            "A single trial checks plumbing, not trained policy performance or sensor calibration.",
            "Wrist wrench is local transmitted wrench; contact force is a separate peg-environment magnitude.",
        ],
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[variant.value for variant in Variant], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    try:
        report, rows = run_probe(Variant(args.variant), args.seed)
        with (args.output / "trace.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        report_path.write_text(json.dumps({
            "schema": "stage-d-shadow-contact-probe-v1", "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
        }, indent=2) + "\n")
        raise
    print(json.dumps({key: report[key] for key in (
        "status", "samples", "peak_contact_force_n", "shadow_policy_calls",
        "shadow_force_observation_calls", "sample_delay_max_s",
    )}))


if __name__ == "__main__":
    main()
