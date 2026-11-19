"""Run one exploratory PPO update in simulation with a frozen nominal policy."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import torch

from envs.peg_in_hole import PegInHoleEnv
from integrations.replay_provenance import replay_provenance
from integrations.residual_experiments import Variant, get_config
from integrations.pi05_droid import Policy, infer_preview
from integrations.residual_ppo_smoke import SmokePolicy, update
from integrations.residual_trajectory import episode_rewards
from integrations.rollout import ResidualRollout, RolloutConfig, run_rollout
from integrations.sim_observation import capture_observation
from run_pi05_preview import LocalOpenPiPolicy
from run_residual_mock import MockNominal


def run(
    variant: Variant, seed: int, episodes: int, config: RolloutConfig, *,
    nominal: Policy | None = None, nominal_metadata: dict | None = None,
) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    experiment = get_config(variant)
    if nominal is None:
        nominal = MockNominal()
        nominal_metadata = {"kind": "mock_zero_action"}
    elif nominal_metadata is None:
        raise ValueError("nominal metadata is required for a supplied policy")
    torch.manual_seed(seed)
    policy = SmokePolicy(force_observation=experiment.force_observation)
    collected = []
    start = time.perf_counter()
    for episode in range(episodes):
        trial_seed = seed + episode
        env = PegInHoleEnv()
        env.reset(seed=trial_seed, lateral_range=0.00005)
        policy.samples.clear()
        with mujoco.Renderer(env.model, height=224, width=224) as renderer:
            result = run_rollout(
                env, renderer, nominal, config,
                residual=ResidualRollout(experiment, policy, trial_seed),
            )
        episode_rewards(result, len(policy.samples))
        collected.append((policy.samples.copy(), result))
    summary = update(policy, collected)
    return {
        "schema": "stage-d-ppo-smoke-v1", "status": "complete",
        "variant": variant.value, "training_seed": seed, "episode_seeds": list(range(seed, seed + episodes)),
        "rollout_config": asdict(config), "replay_provenance": replay_provenance(),
        "nominal_policy": nominal_metadata,
        "torch_version": torch.__version__, "device": "cpu",
        "wall_s": time.perf_counter() - start, "update": summary,
        "limitations": [
            "Single exploratory PPO update; nominal policy is queried but not optimized.",
            "Server metadata is self-reported, not independent proof of checkpoint contents.",
            "No checkpoint, evaluation, comparison, or evidence of useful learning.",
            "Simulation pauses for inference; no real-time or physical-safety claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[variant.value for variant in Variant], default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=0.3)
    parser.add_argument("--nominal", choices=("mock", "local-server"), default="mock")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    args = parser.parse_args()
    if args.seed < 0 or args.episodes < 1 or not 1 <= args.port <= 65535:
        parser.error("seed must be non-negative, episodes positive, and port in 1..65535")
    config = RolloutConfig(timeout_s=args.timeout_s, no_contact=False)
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    try:
        with ExitStack() as stack:
            nominal = None
            metadata = None
            warmup_s = None
            if args.nominal == "local-server":
                nominal = LocalOpenPiPolicy(args.port, response_timeout=180)
                stack.callback(nominal.close)
                env = PegInHoleEnv()
                env.reset(seed=args.seed, lateral_range=0.00005)
                with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                    observation = capture_observation(env, renderer, prompt="Insert the peg into the hole.")
                    start = time.perf_counter()
                    infer_preview(nominal, observation)
                    warmup_s = time.perf_counter() - start
                nominal.response_timeout = config.max_inference_s
                metadata = {"kind": "local_server", "server_metadata": nominal.metadata}
            report = run(Variant(args.variant), args.seed, args.episodes, config,
                         nominal=nominal, nominal_metadata=metadata)
            report["nominal_warmup_wall_s"] = warmup_s
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        report_path.write_text(json.dumps({
            "schema": "stage-d-ppo-smoke-v1", "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
        }, indent=2) + "\n")
        raise
    print(json.dumps({"status": report["status"], "update": report["update"], "report": str(report_path)}))


if __name__ == "__main__":
    main()
