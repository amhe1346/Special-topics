"""Score recorded classical simulation trajectories with the pilot reward."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from benchmark_insertion import TrialConfig, run_trial
from envs.peg_contact import PegContactForce
from integrations.residual_reward import ForceSample, TaskState, interval_reward


def run_probe(
    seed: int, *, timeout_s: float = 12.0, lateral_range_m: float = 0.00005,
    tilt_range_deg: float = 0.03,
) -> dict:
    config = TrialConfig(
        seed=seed, timeout_s=timeout_s, lateral_range_m=lateral_range_m, tilt_range_deg=tilt_range_deg,
    )
    samples = []
    contact = None

    def record(env):
        nonlocal contact
        if contact is None:
            contact = PegContactForce(env)
        geometry = env.measure_geometry()
        samples.append({
            "time_s": float(env.data.time),
            "lateral_error_m": float(geometry.lateral_error),
            "depth_m": float(geometry.depth),
            "contact_force_n": float(contact.measure().magnitude_sum),
        })

    outcome = run_trial("position", config, on_step=record)
    if len(samples) < 2:
        raise RuntimeError("reward probe requires at least two valid physics samples")
    timestep = outcome["timestep_s"]
    start = samples[0]["time_s"]
    previous = TaskState(samples[0]["lateral_error_m"], samples[0]["depth_m"])
    force_samples = []
    intervals = []
    tick = 1
    for index, sample in enumerate(samples[1:], start=1):
        force_samples.append(ForceSample(sample["contact_force_n"], timestep))
        final = index == len(samples) - 1
        if sample["time_s"] + timestep / 2 < start + tick / 15 and not final:
            continue
        state = TaskState(sample["lateral_error_m"], sample["depth_m"])
        termination = outcome["reason"] if final else None
        off = interval_reward(previous, state, force_samples, force_aware=False, termination=termination)
        on = interval_reward(previous, state, force_samples, force_aware=True, termination=termination)
        intervals.append({
            "end_time_s": sample["time_s"],
            "duration_s": len(force_samples) * timestep,
            "reward_off": asdict(off),
            "reward_on": asdict(on),
        })
        previous = state
        force_samples = []
        tick += 1
    return {
        "schema": "stage-d-pilot-reward-probe-v1",
        "status": "complete",
        "outcome": outcome,
        "first_recorded_sample": samples[0],
        "physics_samples": samples,
        "intervals": intervals,
        "undiscounted_total_off": sum(row["reward_off"]["total"] for row in intervals),
        "undiscounted_total_on": sum(row["reward_on"]["total"] for row in intervals),
        "force_excess_n_s": sum(row["reward_on"]["force_excess_n_s"] for row in intervals),
        "limitations": [
            "Classical position controller; no residual training or pi0.5 inference.",
            "First physics step is omitted: callback begins after that step and its initial state is unavailable.",
            "Intervals start at the first recorded sample, not the original rollout's control ticks.",
            "Totals are undiscounted diagnostics, not PPO returns.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--lateral-range-m", type=float, default=0.00005)
    parser.add_argument("--tilt-range-deg", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    try:
        report = run_probe(
            args.seed, timeout_s=args.timeout_s, lateral_range_m=args.lateral_range_m,
            tilt_range_deg=args.tilt_range_deg,
        )
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        report_path.write_text(json.dumps({
            "schema": "stage-d-pilot-reward-probe-v1", "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
        }, indent=2) + "\n")
        raise
    print(json.dumps({
        "reason": report["outcome"]["reason"], "peak_force_n": report["outcome"]["peak_force_n"],
        "force_excess_n_s": report["force_excess_n_s"],
        "total_off": report["undiscounted_total_off"], "total_on": report["undiscounted_total_on"],
        "report": str(report_path),
    }))


if __name__ == "__main__":
    main()
