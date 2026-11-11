import mujoco
import numpy as np
import pytest

from controllers.pd_position_controller import PDPositionController
from envs.peg_in_hole import (
    HOLE_RADIUS,
    MAX_LATERAL_ERROR,
    PEG_RADIUS,
    PegInHoleEnv,
)


def place_hole(env, *, depth, lateral=0.0):
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    env.data.mocap_pos[env.fixture_mocap_id] = tip + [lateral, 0.0, depth]
    mujoco.mj_forward(env.model, env.data)


def test_scene_initializes_above_hole_with_requested_dimensions():
    env = PegInHoleEnv()
    peg_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_peg_geom")

    assert env.model.geom_size[peg_id, 0] == pytest.approx(0.004)
    assert HOLE_RADIUS - PEG_RADIUS == pytest.approx(0.00008)
    assert env.measure_geometry().depth == pytest.approx(-0.015)
    assert not env.measure_geometry().geometric_candidate


def test_reset_randomization_is_seeded_and_resets_simulation():
    env = PegInHoleEnv()
    env.reset(seed=42, lateral_range=0.003)
    first = env.data.mocap_pos[env.fixture_mocap_id].copy()
    env.step(5)
    env.reset(seed=42, lateral_range=0.003)
    np.testing.assert_array_equal(env.data.mocap_pos[env.fixture_mocap_id], first)
    assert env.data.time == 0.0
    env.reset(seed=43, lateral_range=0.003)
    assert not np.array_equal(env.data.mocap_pos[env.fixture_mocap_id], first)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_reset_rejects_invalid_randomization_range(value):
    with pytest.raises(ValueError, match="lateral_range"):
        PegInHoleEnv().reset(lateral_range=value)


@pytest.mark.parametrize(
    ("depth", "lateral", "expected"),
    [
        (0.019, 0.0, False),
        (0.020, 0.0, True),
        (0.022, 0.000079, True),
        (0.022, 0.000080, True),
        (0.022, 0.000081, False),
        (0.025, 0.0, True),
        (0.026, 0.0, False),
    ],
)
def test_geometry_checks_depth_and_lateral_thresholds(depth, lateral, expected):
    env = PegInHoleEnv()
    place_hole(env, depth=depth, lateral=lateral)
    result = env.measure_geometry()
    assert result.depth == pytest.approx(depth)
    assert result.lateral_error == pytest.approx(lateral, abs=1e-12)
    assert result.geometric_candidate is expected


def test_real_collision_geometry_has_clearance_and_blocks_offset_peg():
    env = PegInHoleEnv()
    peg_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_peg_geom")

    def peg_wall_contacts():
        return [
            contact
            for contact in env.data.contact
            if peg_geom in (contact.geom1, contact.geom2)
            and any(
                (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "")
                .startswith("task_hole_wall_")
                for geom in (contact.geom1, contact.geom2)
            )
        ]

    place_hole(env, depth=0.022, lateral=MAX_LATERAL_ERROR * 0.5)
    assert not peg_wall_contacts()

    place_hole(env, depth=0.022, lateral=0.00012)
    assert peg_wall_contacts()
    assert min(contact.dist for contact in peg_wall_contacts()) < 0


def test_tilted_peg_uses_axis_intersection_not_tip_position():
    env = PegInHoleEnv()
    rotation = np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([0.0, 1.0, 0.0]), 0.1)
    peg_quat = env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7].copy()
    result = np.empty(4)
    mujoco.mju_mulQuat(result, rotation, peg_quat)
    env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7] = result
    mujoco.mj_forward(env.model, env.data)
    place_hole(env, depth=0.022)

    measurement = env.measure_geometry()
    assert measurement.angular_error == pytest.approx(0.1)
    assert measurement.lateral_error > MAX_LATERAL_ERROR
    assert not measurement.geometric_candidate


def test_closed_gripper_holds_free_peg_then_opening_drops_it_onto_table():
    env = PegInHoleEnv()
    env.data.mocap_pos[env.fixture_mocap_id, 0] += 0.02
    mujoco.mj_forward(env.model, env.data)
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    initial_tip = env.data.site_xpos[env.peg_tip_id].copy()
    peg_joint_type = env.model.jnt_type[env.peg_joint_id]
    assert peg_joint_type == mujoco.mjtJoint.mjJNT_FREE

    for _ in range(500):
        controller.step(target)
        env.step()
    held_tip = env.data.site_xpos[env.peg_tip_id].copy()
    assert abs(held_tip[2] - initial_tip[2]) < 0.005
    peg_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_peg_geom")
    finger_body_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_finger", "right_finger")
    }
    touching_fingers = set()
    for contact in env.data.contact:
        if contact.geom1 == peg_geom:
            touching_fingers.add(env.model.geom_bodyid[contact.geom2])
        elif contact.geom2 == peg_geom:
            touching_fingers.add(env.model.geom_bodyid[contact.geom1])
    assert finger_body_ids <= touching_fingers

    env.data.ctrl[env.gripper_actuator_id] = 255
    for _ in range(500):
        controller.step(target)
        env.step()
    assert env.data.site_xpos[env.peg_tip_id, 2] < held_tip[2] - 0.02
    table_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_table_geom")
    assert any(
        {int(contact.geom1), int(contact.geom2)} == {peg_geom, table_geom}
        for contact in env.data.contact
    )
    env.reset()
    np.testing.assert_allclose(env.data.site_xpos[env.peg_tip_id], initial_tip, atol=1e-12)
    peg_dof_address = env.model.jnt_dofadr[env.peg_joint_id]
    np.testing.assert_array_equal(env.data.qvel[peg_dof_address : peg_dof_address + 6], 0)
    assert env.data.ctrl[env.gripper_actuator_id] == 0


def test_fixed_workspace_cameras_are_aimed_at_fixture():
    env = PegInHoleEnv()
    target = env.data.site_xpos[env.hole_site_id]
    for name in ("task_left_camera", "task_right_camera"):
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert camera_id >= 0
        forward = -env.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        direction = target - env.data.cam_xpos[camera_id]
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(forward, direction, atol=1e-8)
