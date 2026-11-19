"""Measure local simulation throughput with mock policies, not PPO or pi0.5."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import resource
import time

import mujoco

from envs.peg_in_hole import PegInHoleEnv
from integrations.replay_provenance import replay_provenance
from integrations.residual_experiments import Variant, get_config
from integrations.rollout import ResidualRollout, RolloutConfig, run_rollout
from run_residual_mock import MockNominal, MockResidual


def benchmark(repeats: int, timeout_s: float) -> dict:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    config = RolloutConfig(timeout_s=timeout_s, no_contact=False)
    measurements = []
    warmups = []
    for variant in Variant:
        # A separate warmup is excluded from all measured totals.
        for iteration in range(-1, repeats):
            seed = 0 if iteration < 0 else iteration
            start = time.perf_counter()
            env = PegInHoleEnv()
            env.reset(seed=seed, lateral_range=0.00005)
            policy = MockResidual(0.0)
            environment_ready = time.perf_counter()
            with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                rollout_start = time.perf_counter()
                result = run_rollout(
                    env, renderer, MockNominal(), config,
                    residual=ResidualRollout(get_config(variant), policy, seed),
                )
                rollout_end = time.perf_counter()
            end = time.perf_counter()
            reward = result["residual_experiment"]
            if reward["reward_status"] != "complete":
                raise RuntimeError(f"{variant.value} seed {seed}: invalid rollout: {result['reason']}")
            row = {
                "variant": variant.value,
                "seed": seed,
                "reason": result["reason"],
                "sim_time_s": result["sim_time_s"],
                "physics_steps": result["steps"],
                "action_steps": result["action_steps"],
                "nominal_calls": result["inference_timing"]["calls"],
                "residual_calls": len(policy.observations),
                "environment_setup_wall_s": environment_ready - start,
                "renderer_setup_wall_s": rollout_start - environment_ready,
                "rollout_wall_s": rollout_end - rollout_start,
                "renderer_teardown_wall_s": end - rollout_end,
                "end_to_end_wall_s": end - start,
            }
            (warmups if iteration < 0 else measurements).append(row)
    summaries = []
    for variant in Variant:
        rows = [row for row in measurements if row["variant"] == variant.value]
        wall_s = sum(row["end_to_end_wall_s"] for row in rows)
        summaries.append({
            "variant": variant.value,
            "episodes": len(rows),
            "action_steps": sum(row["action_steps"] for row in rows),
            "physics_steps": sum(row["physics_steps"] for row in rows),
            "sim_time_s": sum(row["sim_time_s"] for row in rows),
            "end_to_end_wall_s": wall_s,
            "action_steps_per_wall_s": sum(row["action_steps"] for row in rows) / wall_s,
            "physics_steps_per_wall_s": sum(row["physics_steps"] for row in rows) / wall_s,
        })
    return {
        "schema": "stage-d-mock-throughput-v1",
        "status": "complete",
        "config": asdict(config),
        "repeats_per_variant": repeats,
        "warmup_episodes_per_variant": 1,
        "nominal_policy": "mock_zero_actions",
        "residual_policy": "mock_zero_actions",
        "render_backend": os.environ.get("MUJOCO_GL", "default"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "max_rss_kib_process_lifetime": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "replay_provenance": replay_provenance(),
        "warmups": warmups,
        "measurements": measurements,
        "summaries": summaries,
        "limitations": [
            "Mock nominal and residual policies: no pi0.5 inference, PPO optimization, or GPU measurement.",
            "End-to-end wall time includes environment reset, renderer setup, grasp settling, and teardown.",
            "Process peak RSS includes all variants and warmups; it is not per-variant memory.",
            "Trials share one local machine and are ordered by variant, not randomized.",
            "Simulation time pauses for inference; throughput is not a real-time guarantee.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the report")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    RolloutConfig(timeout_s=args.timeout_s, no_contact=False)
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    try:
        report = benchmark(args.repeats, args.timeout_s)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        report_path.write_text(json.dumps({
            "schema": "stage-d-mock-throughput-v1", "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
        }, indent=2) + "\n")
        raise
    print(json.dumps({"status": report["status"], "summaries": report["summaries"], "report": str(report_path)}))


if __name__ == "__main__":
    main()
