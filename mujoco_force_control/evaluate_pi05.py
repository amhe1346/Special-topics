"""Evaluate one local policy on the oracle dataset's simulation conditions."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from collect_scripted_teacher import _set_fixture_tilt
from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import infer_preview
from integrations.replay_provenance import replay_provenance
from integrations.rollout import RolloutConfig, run_rollout
from integrations.sim_observation import capture_observation
from run_pi05_preview import LocalOpenPiPolicy


PROMPT = "Insert the peg into the hole."


def _symmetric_range(value: list[float], name: str) -> float:
    if len(value) != 2 or not all(np.isfinite(v) for v in value):
        raise ValueError(f"{name} must be a finite symmetric range")
    if value[1] < 0 or not np.isclose(value[0], -value[1], rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be a non-negative symmetric range")
    return float(value[1])


def _new_environment(seed: int, lateral_range_m: float, tilt_range_deg: float) -> PegInHoleEnv:
    env = PegInHoleEnv()
    env.reset(seed=seed, lateral_range=lateral_range_m)
    _set_fixture_tilt(env, seed, tilt_range_deg)
    return env


def evaluate(
    policy,
    source_manifest: Path,
    output: Path,
    *,
    split: str = "validation",
    config: RolloutConfig = RolloutConfig(timeout_s=12.0, no_contact=False),
    repeats: int = 1,
    warmup: bool = True,
    on_warmup: Callable[[], None] | None = None,
) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    source_bytes = source_manifest.read_bytes()
    source = json.loads(source_bytes)
    if source.get("schema") != "stage-c-oracle-v2" or source.get("status") != "complete":
        raise ValueError("expected a complete stage-c-oracle-v2 manifest")
    seeds = source["splits"][split]
    all_seeds = [seed for group in source["splits"].values() for seed in group]
    if (
        not seeds or any(type(seed) is not int or seed < 0 for seed in all_seeds)
        or len(all_seeds) != len(set(all_seeds))
    ):
        raise ValueError("source splits must contain disjoint non-negative integer seeds")
    randomization = source["randomization"]
    lateral = _symmetric_range(randomization["lateral_offset_range_m"], "lateral offset")
    tilt = _symmetric_range(randomization["fixture_tilt_xy_range_deg"], "fixture tilt")
    if tilt >= 90:
        raise ValueError("fixture tilt range must be less than 90 degrees")

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {
        "schema": "stage-c-policy-evaluation-v1",
        "status": "in_progress",
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "split": split,
        "seeds": seeds,
        "repeats": repeats,
        "randomization": randomization,
        "rollout_config": asdict(config),
        "prompt": PROMPT,
        "policy_metadata": policy.metadata,
        "replay_provenance": replay_provenance(),
        "warmup": None,
        "trials": [],
        "limitations": [
            "Simulation only; inference pauses the simulation and does not establish real-time control or hardware safety.",
            "Evaluation uses open-loop chunks of eight by default; oracle demonstrations replanned every action.",
            "Policy sampling is stochastic and its RNG advances across warmup and trials.",
            "Repeated fixture seeds measure within-server variation; policy RNG is not reset between trials.",
            "Server metadata identifies its configured checkpoint but is not independent proof of checkpoint contents.",
        ],
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if warmup:
            env = _new_environment(seeds[0], lateral, tilt)
            with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                observation = capture_observation(env, renderer, prompt=PROMPT)
                start = time.monotonic()
                infer_preview(policy, observation)
                report["warmup"] = {"inference_wall_s": time.monotonic() - start}
            if on_warmup is not None:
                on_warmup()
            save()
        for repeat in range(repeats):
            for seed in seeds:
                env = _new_environment(seed, lateral, tilt)
                fixture_position = env.data.mocap_pos[env.fixture_mocap_id].tolist()
                fixture_quaternion = env.data.mocap_quat[env.fixture_mocap_id].tolist()
                with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                    result = run_rollout(env, renderer, policy, config, prompt=PROMPT)
                geometry = None if result["reason"] == "invalid_state" else env.measure_geometry()
                report["trials"].append({
                    "seed": seed,
                    "repeat": repeat,
                    "fixture_position": fixture_position,
                    "fixture_quaternion": fixture_quaternion,
                    "lateral_error_m": None if geometry is None else geometry.lateral_error,
                    "angular_error_rad": None if geometry is None else geometry.angular_error,
                    "result": result,
                })
                save()
        report["summary"] = {
            "trials": len(report["trials"]),
            "successes": sum(trial["result"]["reason"] == "success" for trial in report["trials"]),
            "outcomes": {
                reason: sum(trial["result"]["reason"] == reason for trial in report["trials"])
                for reason in sorted({trial["result"]["reason"] for trial in report["trials"]})
            },
            "per_seed": [
                {
                    "seed": seed,
                    "successes": sum(
                        trial["result"]["reason"] == "success"
                        for trial in report["trials"] if trial["seed"] == seed
                    ),
                    "outcomes": [
                        trial["result"]["reason"]
                        for trial in report["trials"] if trial["seed"] == seed
                    ],
                    "peak_forces_n": [
                        trial["result"]["peak_force_n"]
                        for trial in report["trials"] if trial["seed"] == seed
                    ],
                }
                for seed in seeds
            ],
        }
        report["status"] = "complete"
        save()
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--source-manifest", type=Path, default=Path("results/stage_c_oracle_v2/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    policy = LocalOpenPiPolicy(args.port, response_timeout=180.0)
    try:
        if policy.metadata.get("evaluation_server") != "stage-c-policy-server-v1":
            raise ValueError("server must be started by run_openpi_local_server.py")
        report = evaluate(
            policy, args.source_manifest, args.output, split=args.split, repeats=args.repeats,
            on_warmup=lambda: setattr(policy, "response_timeout", 5.0),
        )
    finally:
        policy.close()
    print(json.dumps(report["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
