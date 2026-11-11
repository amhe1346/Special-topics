"""Capture a simulated observation and preview policy actions without moving the robot."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json

import mujoco
import numpy as np

from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import CHUNK_SHAPE, infer_preview
from integrations.sim_observation import capture_observation


class ZeroPolicy:
    """Test the adapter without an installed model."""

    def infer(self, observation: dict) -> dict:
        return {"actions": np.zeros(CHUNK_SHAPE)}


class LocalOpenPiPolicy:
    """Local-only OpenPI websocket client with bounded connection and response waits."""

    def __init__(self, port: int, response_timeout: float = 180.0) -> None:
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as error:
            raise RuntimeError("Install the official openpi-client before using --policy local-server") from error
        self.msgpack = msgpack_numpy
        if not np.isfinite(response_timeout) or response_timeout <= 0:
            raise ValueError("response_timeout must be finite and positive")
        self.response_timeout = response_timeout
        with ExitStack() as stack:
            self.connection = stack.enter_context(
                connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None,
                        open_timeout=5, close_timeout=2)
            )
            metadata = self.connection.recv(timeout=30)
            if not isinstance(metadata, bytes) or not isinstance(self.msgpack.unpackb(metadata), dict):
                raise ValueError("OpenPI server returned invalid metadata")
            self._connection_context = stack.pop_all()

    def infer(self, observation: dict) -> dict:
        self.connection.send(self.msgpack.packb(observation))
        response = self.connection.recv(timeout=self.response_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server error: {response}")
        result = self.msgpack.unpackb(response)
        if not isinstance(result, dict):
            raise ValueError("OpenPI server returned a non-dictionary response")
        return result

    def close(self) -> None:
        self._connection_context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("mock", "local-server"), default="mock")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", choices=("left", "right"), default="left")
    parser.add_argument("--prompt", default="Insert the peg into the hole.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    env = PegInHoleEnv()
    env.reset(seed=args.seed)
    with mujoco.Renderer(env.model, height=224, width=224) as renderer:
        observation = capture_observation(
            env, renderer, prompt=args.prompt, exterior_camera=f"task_{args.camera}_camera"
        )
    with ExitStack() as stack:
        if args.policy == "local-server":
            policy = LocalOpenPiPolicy(args.port)
            stack.callback(policy.close)
        else:
            policy = ZeroPolicy()
        actions = infer_preview(policy, observation)
    print(json.dumps({
        "policy": args.policy,
        "checkpoint_verified_by_server": False,
        "simulation_time_s": observation.time_s,
        "exterior_camera": f"task_{args.camera}_camera",
        "wrist_camera": "task_wrist_camera",
        "gripper_closed_fraction": observation.gripper_closed_fraction,
        "action_count": len(actions),
        "first_normalized_joint_action": actions[0].normalized_velocity.tolist(),
        "first_suggested_joint_delta_rad": actions[0].joint_delta_rad.tolist(),
        "first_gripper_closed": actions[0].gripper_closed,
        "actuated": False,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
