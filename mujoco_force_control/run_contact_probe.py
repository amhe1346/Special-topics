"""Measure contact forces in provisional, scripted simulation scenarios."""

from __future__ import annotations

import argparse
import csv
import sys

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from envs.peg_contact import PegContactForce
from envs.peg_in_hole import HOLE_DEPTH, PegInHoleEnv


def run_probe(scenario: str, speed: float) -> list[tuple[float, float, float, float, float, float]]:
    """Raise the fixture toward a held peg and record time, depth and force."""
    if scenario not in ("aligned", "jam", "press"):
        raise ValueError("scenario must be aligned, jam, or press")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    env = PegInHoleEnv()
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    # Settle the grasp with the fixture out of the way.
    env.data.mocap_pos[env.fixture_mocap_id, 0] += 0.1
    mujoco.mj_forward(env.model, env.data)
    for _ in range(250):
        controller.step(target)
        env.step()
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    lateral = 0.0 if scenario != "jam" else 0.003
    entrance_z = tip[2] - 0.005
    env.data.mocap_pos[env.fixture_mocap_id] = (tip[0] + lateral, tip[1], entrance_z)
    env.data.mocap_pos[env.table_mocap_id, 2] = entrance_z - HOLE_DEPTH
    mujoco.mj_forward(env.model, env.data)
    contact = PegContactForce(env)
    travel = 0.042 if scenario == "press" else 0.028
    count = int(np.ceil(travel / (speed * env.model.opt.timestep)))
    rows = []
    for _ in range(count):
        env.data.mocap_pos[env.fixture_mocap_id, 2] += speed * env.model.opt.timestep
        env.data.mocap_pos[env.table_mocap_id, 2] += speed * env.model.opt.timestep
        controller.step(target)
        env.step()
        mujoco.mj_forward(env.model, env.data)
        force = contact.measure()
        rows.append((float(env.data.time), env.measure_geometry().depth, force.magnitude_sum, *force.world_vector))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.01, help="fixture speed in m/s")
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(("scenario", "speed_m_s", "lateral_offset_m", "time_s", "depth_m", "contact_force_n", "force_x_n", "force_y_n", "force_z_n"))
    for scenario in ("aligned", "jam", "press"):
        for row in run_probe(scenario, args.speed):
            writer.writerow((scenario, args.speed, 0.003 if scenario == "jam" else 0.0, *row))


if __name__ == "__main__":
    main()
