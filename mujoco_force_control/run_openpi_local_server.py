"""Serve the official pi05_droid checkpoint on localhost only."""

import argparse
import logging

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO)
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_droid"),
        "gs://openpi-assets/checkpoints/pi05_droid",
    )
    WebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
