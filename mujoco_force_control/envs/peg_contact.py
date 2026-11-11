"""Ground-truth peg contact-force measurements."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from envs.peg_in_hole import WALL_COUNT, PegInHoleEnv


@dataclass(frozen=True)
class ContactForce:
    magnitude_sum: float
    world_vector: tuple[float, float, float]
    contact_count: int = 0


class PegContactForce:
    """Sum force magnitudes for peg contacts with the fixture and table."""

    def __init__(self, env: PegInHoleEnv) -> None:
        self.env = env
        self.peg_geom = env._named_id(mujoco.mjtObj.mjOBJ_GEOM, "task_peg_geom")
        self.surface_geoms = {
            env._named_id(mujoco.mjtObj.mjOBJ_GEOM, "task_table_geom"),
            *(
                env._named_id(mujoco.mjtObj.mjOBJ_GEOM, f"task_hole_wall_{i}")
                for i in range(WALL_COUNT)
            ),
        }

    def measure(self, *, surface_only: bool = False) -> ContactForce:
        """Return summed magnitudes and net world force on the peg, in newtons."""
        wrench = np.zeros(6)
        total = 0.0
        vector = np.zeros(3)
        count = 0
        for index, contact in enumerate(self.env.data.contact):
            geoms = {int(contact.geom1), int(contact.geom2)}
            if self.peg_geom in geoms and geoms.intersection(self.surface_geoms):
                if surface_only:
                    if contact.dist > 0:
                        continue
                    rotation = self.env.data.site_xmat[self.env.hole_site_id].reshape(3, 3)
                    position = rotation.T @ (contact.pos - self.env.data.site_xpos[self.env.hole_site_id])
                    if abs(position[2]) > 0.001:
                        continue
                mujoco.mj_contactForce(self.env.model, self.env.data, index, wrench)
                total += float(np.linalg.norm(wrench[:3]))
                sign = 1.0 if int(contact.geom2) == self.peg_geom else -1.0
                vector += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
                count += 1
        if not np.isfinite(total) or not np.all(np.isfinite(vector)):
            raise ValueError("Non-finite peg contact force")
        return ContactForce(total, (float(vector[0]), float(vector[1]), float(vector[2])), count)
