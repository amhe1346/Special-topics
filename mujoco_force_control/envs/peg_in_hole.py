"""Initial peg-in-hole scene and geometric measurements for the Panda."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from envs.base_env import MODEL_PATH, BaseEnv


PEG_RADIUS = 0.004
HOLE_RADIUS = 0.00408
PEG_LENGTH = 0.06
HOLE_DEPTH = 0.03
MIN_DEPTH = 0.020
MAX_DEPTH = 0.025
MAX_LATERAL_ERROR = 0.00008
WALL_COUNT = 64
GEOMETRY_ROUNDOFF = 1e-12
PEG_HAND_OFFSET = 0.105
INITIAL_FINGER_POSITION = 0.003


@dataclass(frozen=True)
class PegGeometry:
    """True simulation geometry, measured in metres and radians."""

    depth: float
    lateral_error: float
    angular_error: float
    axis_intersects_entrance: bool
    depth_in_range: bool
    lateral_in_range: bool

    @property
    def geometric_candidate(self) -> bool:
        """Return the geometric checks, not a complete task success decision."""
        return (
            self.axis_intersects_entrance
            and self.depth_in_range
            and self.lateral_in_range
        )


def _scene_xml(model_path: Path) -> str:
    """Add task-only geometry to the existing Panda without changing its MJCF."""
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    world = root.find("worldbody")
    hand = root.find(".//body[@name='hand']")
    if compiler is None or world is None or hand is None:
        raise ValueError("Panda model is missing its compiler, worldbody, or hand")
    meshdir = compiler.get("meshdir", "")
    compiler.set("meshdir", str((model_path.parent / meshdir).resolve()))
    ET.SubElement(
        hand, "camera", name="task_wrist_camera", pos="0.06 0 0.025",
        xyaxes="0 1 0 0.877896 0 0.478852", fovy="70",
    )

    peg = ET.SubElement(world, "body", name="task_peg", pos="0.55 0 0.5")
    ET.SubElement(peg, "freejoint", name="task_peg_free")
    ET.SubElement(
        peg, "geom", name="task_peg_geom", type="cylinder",
        size=f"{PEG_RADIUS} {PEG_LENGTH / 2}", mass="0.03",
        rgba="0.2 0.45 0.85 1", contype="2", conaffinity="5",
        friction="1 0.005 0.0001",
    )
    ET.SubElement(peg, "site", name="task_peg_tip", pos=f"0 0 {PEG_LENGTH / 2}", size="0.002")
    ET.SubElement(peg, "site", name="task_peg_base", pos=f"0 0 {-PEG_LENGTH / 2}", size="0.002")

    fixture = ET.SubElement(world, "body", name="task_fixture", mocap="true", pos="0.55 0 0.45")
    ET.SubElement(fixture, "site", name="task_hole_entrance", size="0.002", rgba="0 1 0 0.4")
    # Tangent boxes approximate the circular inner wall without filling the hole.
    for index in range(WALL_COUNT):
        angle = 2 * np.pi * index / WALL_COUNT
        radius = HOLE_RADIUS + 0.006
        ET.SubElement(
            fixture, "geom", name=f"task_hole_wall_{index}", type="box",
            pos=f"{radius * np.cos(angle):.12g} {radius * np.sin(angle):.12g} {-HOLE_DEPTH / 2}",
            euler=f"0 0 {angle:.12g}", size=f"0.006 0.002 {HOLE_DEPTH / 2}",
            rgba="0.1 0.7 0.2 1", contype="4", conaffinity="3",
            friction="0.5 0.005 0.0001",
        )
    table = ET.SubElement(world, "body", name="task_table", mocap="true", pos="0.55 0 0.42")
    ET.SubElement(
        table, "geom", name="task_table_geom", type="box",
        pos="0 0 -0.02", size="0.25 0.25 0.02",
        rgba="0.45 0.45 0.45 1", contype="1", conaffinity="3",
        friction="1 0.005 0.0001",
    )
    for name, pos in (
        ("task_left_camera", "0.85 0.45 0.75"),
        ("task_right_camera", "0.85 -0.45 0.75"),
    ):
        ET.SubElement(world, "camera", name=name, pos=pos, mode="targetbody", target="task_fixture")
    for key in root.findall("./keyframe/key"):
        key.set("qpos", f"{key.get('qpos', '')} 0.55 0 0.5 1 0 0 0")
    return ET.tostring(root, encoding="unicode")


class PegInHoleEnv(BaseEnv):
    """Panda with a free peg, gripper contact, table and movable hole."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.peg_tip_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "task_peg_tip")
        self.peg_base_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "task_peg_base")
        self.hole_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "task_hole_entrance")
        fixture_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "task_fixture")
        self.fixture_mocap_id = int(self.model.body_mocapid[fixture_id])
        table_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "task_table")
        self.table_mocap_id = int(self.model.body_mocapid[table_id])
        self.hand_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.peg_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "task_peg_free")
        self.peg_qpos_address = int(self.model.jnt_qposadr[self.peg_joint_id])
        self.finger_joint_ids = [
            self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("finger_joint1", "finger_joint2")
        ]
        self.gripper_actuator_id = self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.reset()

    def _named_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, obj_type, name)
        if identifier < 0:
            raise KeyError(f"{name!r} is not defined in the task model")
        return identifier

    def reset(
        self,
        keyframe: str | None = "home",
        *,
        seed: int | None = None,
        lateral_range: float = 0.0,
    ) -> None:
        """Reset with the free peg between the fingers and the hole below it."""
        if not np.isfinite(lateral_range) or lateral_range < 0:
            raise ValueError("lateral_range must be finite and non-negative")
        super().reset(keyframe)
        self.data.qpos[self.model.jnt_qposadr[self.finger_joint_ids]] = INITIAL_FINGER_POSITION
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        hand_rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        self.data.qpos[self.peg_qpos_address : self.peg_qpos_address + 3] = (
            self.data.xpos[self.hand_id] + hand_rotation @ np.array([0.0, 0.0, PEG_HAND_OFFSET])
        )
        self.data.qpos[self.peg_qpos_address + 3 : self.peg_qpos_address + 7] = (
            self.data.xquat[self.hand_id]
        )
        mujoco.mj_forward(self.model, self.data)
        tip = self.data.site_xpos[self.peg_tip_id].copy()
        offset = np.random.default_rng(seed).uniform(-lateral_range, lateral_range, 2)
        self.data.mocap_pos[self.fixture_mocap_id] = (
            tip[0] + offset[0], tip[1] + offset[1], tip[2] - 0.015
        )
        self.data.mocap_quat[self.fixture_mocap_id] = (1, 0, 0, 0)
        self.data.mocap_pos[self.table_mocap_id] = (tip[0], tip[1], tip[2] - 0.015 - HOLE_DEPTH)
        self.data.mocap_quat[self.table_mocap_id] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)

    def measure_geometry(self) -> PegGeometry:
        """Measure peg depth and axis alignment relative to the hole entrance."""
        tip = self.data.site_xpos[self.peg_tip_id].copy()
        base = self.data.site_xpos[self.peg_base_id].copy()
        entrance = self.data.site_xpos[self.hole_site_id].copy()
        rotation = self.data.site_xmat[self.hole_site_id].reshape(3, 3)
        tip_local = rotation.T @ (tip - entrance)
        base_local = rotation.T @ (base - entrance)
        axis = tip_local - base_local
        axis_length = float(np.linalg.norm(axis))
        depth = float(-tip_local[2])
        cosine = float(np.clip(-axis[2] / axis_length, -1.0, 1.0))
        angular_error = float(np.arccos(cosine))
        crosses = bool(base_local[2] >= 0 and tip_local[2] <= 0 and axis[2] < -1e-12)
        if crosses:
            fraction = -base_local[2] / axis[2]
            lateral_error = float(np.linalg.norm((base_local + fraction * axis)[:2]))
        else:
            lateral_error = float(np.linalg.norm(tip_local[:2]))
        return PegGeometry(
            depth=depth,
            lateral_error=lateral_error,
            angular_error=angular_error,
            axis_intersects_entrance=crosses,
            depth_in_range=MIN_DEPTH - GEOMETRY_ROUNDOFF <= depth <= MAX_DEPTH + GEOMETRY_ROUNDOFF,
            lateral_in_range=lateral_error <= MAX_LATERAL_ERROR + GEOMETRY_ROUNDOFF,
        )
