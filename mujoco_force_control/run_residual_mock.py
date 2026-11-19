"""Run a simulation-only plumbing check with mock nominal and residual policies."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from envs.peg_in_hole import PegInHoleEnv
from integrations.replay_provenance import replay_provenance
from integrations.residual_adapter import ForceResidualObservation
from integrations.residual_experiments import Variant, get_config
from integrations.rollout import ResidualRollout, RolloutConfig, run_rollout


class MockNominal:
    def infer(self, observation):
        actions = np.zeros((15, 8))
        actions[:, 7] = 1.0
        return {"actions": actions}


class MockResidual:
    def __init__(self, correction_rad: float):
        self.correction_rad = correction_rad
        self.observations = []

    def infer(self, observation):
        self.observations.append(isinstance(observation, ForceResidualObservation))
        return np.full(7, self.correction_rad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[variant.value for variant in Variant], default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--correction-rad", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the mock report")
    args = parser.parse_args()
    if args.seed < 0 or not np.isfinite(args.correction_rad):
        parser.error("seed must be non-negative and correction must be finite")
    config = RolloutConfig(timeout_s=args.timeout_s, no_contact=False)
    experiment = get_config(Variant(args.variant))
    env = PegInHoleEnv()
    env.reset(seed=args.seed, lateral_range=0.00005)
    residual_policy = MockResidual(args.correction_rad)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "stage-d-mock-rollout-v1",
        "status": "in_progress",
        "nominal_policy": "mock_zero_actions",
        "residual_policy": "mock_constant",
        "correction_rad": args.correction_rad,
        "seed": args.seed,
        "config": asdict(config),
        "experiment": asdict(experiment),
        "replay_provenance": replay_provenance(),
        "limitations": [
            "Simulation-only plumbing test; neither policy is trained.",
            "Pilot reward accounting only; no training is implemented.",
            "Synchronous inference pauses simulation; no real-time or physical safety claim.",
        ],
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        with mujoco.Renderer(env.model, height=224, width=224) as renderer:
            result = run_rollout(
                env, renderer, MockNominal(), config,
                residual=ResidualRollout(experiment, residual_policy, args.seed),
            )
        report.update(
            status="complete", outcome=result,
            residual_calls=len(residual_policy.observations),
            force_observations=sum(residual_policy.observations),
        )
    except Exception as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"], "reason": result["reason"], "steps": result["steps"],
        "action_steps": result["action_steps"], "residual_calls": report["residual_calls"],
        "force_observations": report["force_observations"], "report": str(report_path),
    }))


if __name__ == "__main__":
    main()
