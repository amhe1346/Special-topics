"""Run guarded pi0.5 policy control in MuJoCo, never on a physical robot."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from collect_stage_c import Recorder
from envs.peg_in_hole import PegInHoleEnv
from integrations.replay_provenance import replay_provenance
from integrations.rollout import RolloutConfig, run_rollout
from run_pi05_preview import LocalOpenPiPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=6.0)
    parser.add_argument("--max-inference-s", type=float, default=5.0)
    parser.add_argument(
        "--open-loop-horizon", type=int, default=8,
        help="actions to execute from each predicted chunk before re-querying (max 15)",
    )
    parser.add_argument("--allow-contact", action="store_true", help="permit guarded simulation contact")
    parser.add_argument("--output", type=Path, help="new directory for recorded simulation trial")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.seed < 0:
        parser.error("port must be 1..65535 and seed must be non-negative")
    env = PegInHoleEnv()
    env.reset(seed=args.seed, lateral_range=0.00005)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder() if args.output is not None else None
    observations = []
    config = RolloutConfig(
        timeout_s=args.timeout_s, max_inference_s=args.max_inference_s,
        open_loop_horizon=args.open_loop_horizon, no_contact=not args.allow_contact,
    )
    metadata = {"schema": "stage-c-development-v2", "source": "pi05_droid",
                "seed": args.seed, "config": asdict(config), "status": "in_progress",
                "replay_provenance": replay_provenance()}
    if args.output is not None:
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    policy = None
    try:
        policy = LocalOpenPiPolicy(args.port, response_timeout=config.max_inference_s)
        with mujoco.Renderer(env.model, height=224, width=224) as renderer:
            result = run_rollout(env, renderer, policy, config, on_step=recorder,
                                 on_observation=observations.append if recorder is not None else None)
        metadata.update(status="complete", outcome=result)
    except Exception as error:
        metadata.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if policy is not None:
            policy.close()
        if recorder is not None:
            recorder.close()
        if args.output is not None:
            if recorder is not None:
                recorder.save(args.output / "samples.npz", args.output / "steps.npz")
            if observations:
                requests = [observation.request() for observation in observations]
                np.savez_compressed(
                    args.output / "observations.npz",
                    time_s=np.asarray([observation.time_s for observation in observations]),
                    exterior_rgb=np.asarray([request["observation/exterior_image_1_left"] for request in requests]),
                    wrist_rgb=np.asarray([request["observation/wrist_image_left"] for request in requests]),
                    joint_position_rad=np.asarray([request["observation/joint_position"] for request in requests]),
                    gripper_closed_fraction=np.asarray([observation.gripper_closed_fraction for observation in observations]),
                    prompt=np.asarray([observation.prompt for observation in observations]),
                )
            (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
