"""Shared MuJoCo simulation wrapper for the control learning stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(__file__).parent / "assets" / "panda" / "panda.xml"


class BaseEnv:
    """Load a MuJoCo model and expose stepping and named sensor reads."""

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, keyframe: str | None = None) -> None:
        """Return the simulation to its initial state."""
        if keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            keyframe_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
            )
            if keyframe_id < 0:
                raise KeyError(f"Keyframe {keyframe!r} is not defined in the model")
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by ``n_steps`` physics steps."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def read_sensor(self, name: str) -> np.ndarray:
        """Return a copy of a named sensor's values."""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise KeyError(f"Sensor {name!r} is not defined in the model")
        address = self.model.sensor_adr[sensor_id]
        dimension = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[address : address + dimension].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer", action="store_true", help="open the interactive MuJoCo viewer"
    )
    args = parser.parse_args()

    env = BaseEnv()
    if args.viewer:
        mujoco.viewer.launch(env.model, env.data)
    else:
        env.step(10)
        print(
            f"Loaded {MODEL_PATH.name}: {env.model.nq} position coordinates, "
            f"{env.model.nu} actuators, time={env.data.time:.3f}s"
        )


if __name__ == "__main__":
    main()