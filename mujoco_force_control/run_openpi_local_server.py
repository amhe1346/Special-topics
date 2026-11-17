"""Serve a pi05 DROID checkpoint on localhost for simulation only."""

import argparse
import logging
from pathlib import Path
import subprocess

import numpy as np

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"


class HorizonAdapter:
    """Expose the first 15 actions of a 16-step fine-tuned model."""

    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"])
        if actions.shape != (16, 8) or not np.all(np.isfinite(actions)):
            raise ValueError("fine-tuned policy must return finite (16, 8) actions")
        return {**result, "actions": actions[:15]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="local fine-tuned checkpoint step directory containing params and assets",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.checkpoint is None:
        config_name = "pi05_droid"
        checkpoint = BASE_CHECKPOINT
        native_horizon = 15
    else:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not (checkpoint_path / "params").is_dir():
            parser.error("fine-tuned checkpoint must contain a params directory")
        if not (checkpoint_path / "assets/droid/norm_stats.json").is_file():
            parser.error("fine-tuned checkpoint must contain assets/droid/norm_stats.json")
        config_name = "pi05_droid_finetune"
        checkpoint = str(checkpoint_path)
        native_horizon = 16
    logging.basicConfig(level=logging.INFO)
    policy = policy_config.create_trained_policy(
        config.get_config(config_name), checkpoint,
    )
    openpi_root = Path(__file__).resolve().parent / ".openpi"
    revision = subprocess.run(
        ["git", "-C", str(openpi_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    metadata = {
        **policy.metadata,
        "evaluation_server": "stage-c-policy-server-v1",
        "config": config_name,
        "checkpoint": checkpoint,
        "openpi_revision": revision,
        "native_action_horizon": native_horizon,
        "served_action_horizon": 15,
    }
    WebsocketPolicyServer(
        policy=HorizonAdapter(policy) if native_horizon == 16 else policy,
        host="127.0.0.1", port=args.port, metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
