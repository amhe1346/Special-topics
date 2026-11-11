import mujoco
import numpy as np
import pytest

from controllers.pd_position_controller import PDPositionController
from envs.peg_contact import ContactForce, PegContactForce
from envs.peg_episode import EpisodeConfig, PegEpisode
from envs.peg_in_hole import HOLE_DEPTH, PegInHoleEnv


def place_hole(env, depth, lateral=0.0):
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    env.data.mocap_pos[env.fixture_mocap_id] = tip + [lateral, 0.0, depth]
    mujoco.mj_forward(env.model, env.data)


def frozen_step(env):
    env.data.time += env.model.opt.timestep


def test_continuous_dwell_resets_and_requires_actual_insertion(monkeypatch):
    env = PegInHoleEnv()
    monkeypatch.setattr(env, "step", lambda: frozen_step(env))
    episode = PegEpisode(env)
    for _ in range(10):
        assert episode.step() is None
    assert episode.dwell_s == 0
    place_hole(env, 0.022)
    for _ in range(49):
        assert episode.step() is None
    place_hole(env, 0.019)
    assert episode.step() is None
    assert episode.dwell_s == 0
    place_hole(env, 0.022)
    for _ in range(49):
        assert episode.step() is None
    assert episode.step() == "success"
    with pytest.raises(RuntimeError, match="terminated"):
        episode.step()


def test_tilt_and_inserted_portion_clearance_both_matter():
    env = PegInHoleEnv()
    angle = np.deg2rad(0.19)
    rotation = np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([0.0, 1.0, 0.0]), angle)
    quat = env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7].copy()
    result = np.empty(4)
    mujoco.mju_mulQuat(result, rotation, quat)
    env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7] = result
    mujoco.mj_forward(env.model, env.data)
    place_hole(env, 0.022, lateral=0.0001)
    episode = PegEpisode(env)
    assert env.measure_geometry().geometric_candidate
    assert not episode._seated()
    angle = np.deg2rad(0.21)
    mujoco.mju_axisAngle2Quat(rotation, np.array([0.0, 1.0, 0.0]), angle)
    mujoco.mju_mulQuat(result, rotation, quat)
    env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7] = result
    mujoco.mj_forward(env.model, env.data)
    assert not episode._seated()


def test_timeout_invalid_state_and_workspace(monkeypatch):
    env = PegInHoleEnv()
    monkeypatch.setattr(env, "step", lambda: frozen_step(env))
    episode = PegEpisode(env, EpisodeConfig(timeout_s=0.004))
    assert episode.step() is None
    assert episode.step() == "timeout"
    env.reset()
    episode = PegEpisode(env)
    env.data.qvel[0] = np.nan
    assert episode.step() == "invalid_state"
    env.reset()
    episode = PegEpisode(env, EpisodeConfig(workspace_xy_m=0.001))
    env.data.qpos[env.peg_qpos_address] += 0.01
    assert episode.step() == "workspace"


def test_force_stop_precedes_success_and_records_interval(monkeypatch):
    env = PegInHoleEnv()
    monkeypatch.setattr(env, "step", lambda: frozen_step(env))
    place_hole(env, 0.022)
    episode = PegEpisode(env, EpisodeConfig(dwell_s=0.002))
    monkeypatch.setattr(episode.contact, "measure", lambda: ContactForce(6.0, (0.0, 0.0, 6.0)))
    assert episode.step() == "force_limit"
    assert episode.samples[0].seated
    assert episode.peak_force_n > episode.config.hard_force_n
    assert episode.above_reference_intervals == [(0.0, pytest.approx(0.002))]
    assert len(episode.samples) == 1


def test_reference_intervals_end_when_force_falls(monkeypatch):
    env = PegInHoleEnv()
    monkeypatch.setattr(env, "step", lambda: frozen_step(env))
    episode = PegEpisode(env, EpisodeConfig(timeout_s=0.008))
    forces = iter((0.5, 2.0, 2.0, 0.5))
    monkeypatch.setattr(episode.contact, "measure", lambda: ContactForce(next(forces), (0.0, 0.0, 0.0)))
    for _ in range(3):
        assert episode.step() is None
    assert episode.step() == "timeout"
    assert episode.above_reference_intervals == [(pytest.approx(0.002), pytest.approx(0.006))]


def test_sustained_loss_of_finger_contact_is_a_drop_without_large_displacement(monkeypatch):
    env = PegInHoleEnv()
    monkeypatch.setattr(env, "step", lambda: frozen_step(env))
    episode = PegEpisode(env)
    monkeypatch.setattr(episode, "_touching_finger", lambda: False)
    for _ in range(49):
        assert episode.step() is None
    assert episode.step() == "dropped_peg"
    assert np.linalg.norm(episode._hand_offset() - episode.initial_hand_offset) < 1e-12


@pytest.mark.parametrize("kwargs", [
    {"dwell_s": 0.0}, {"hard_force_n": float("nan")},
    {"reference_force_n": 5.0}, {"angle_rad": np.pi},
])
def test_invalid_episode_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EpisodeConfig(**kwargs)


@pytest.mark.parametrize("intended", [False, True])
def test_real_gripper_release_is_only_a_drop_without_intent(intended):
    env = PegInHoleEnv()
    env.data.mocap_pos[env.fixture_mocap_id, 0] += 0.02
    mujoco.mj_forward(env.model, env.data)
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    for _ in range(500):
        controller.step(target)
        env.step()
    episode = PegEpisode(env, EpisodeConfig(timeout_s=1.0))
    episode.set_release_intent(intended)
    env.data.ctrl[env.gripper_actuator_id] = 255
    for _ in range(500):
        controller.step(target)
        reason = episode.step()
        if reason is not None:
            break
    assert reason == ("timeout" if intended else "dropped_peg")
    if intended:
        assert np.linalg.norm(episode._hand_offset() - episode.initial_hand_offset) > episode.config.drop_distance_m
    else:
        assert (episode._no_grasp_s >= episode.config.grasp_loss_s - 1e-12
                or np.linalg.norm(episode._hand_offset() - episode.initial_hand_offset) > episode.config.drop_distance_m)


def test_table_contact_force_points_up_on_peg():
    env = PegInHoleEnv()
    tip = env.data.site_xpos[env.peg_tip_id]
    env.data.mocap_pos[env.table_mocap_id, 2] = tip[2] + 0.001
    mujoco.mj_forward(env.model, env.data)
    force = PegContactForce(env).measure()
    assert force.magnitude_sum > 0
    assert force.world_vector[2] > 0


def test_real_controlled_insertion_reaches_success():
    env = PegInHoleEnv()
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    for _ in range(250):
        controller.step(target)
        env.step()
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    entrance_z = tip[2] - 0.005
    env.data.mocap_pos[env.fixture_mocap_id] = (tip[0], tip[1], entrance_z)
    env.data.mocap_pos[env.table_mocap_id, 2] = entrance_z - HOLE_DEPTH
    mujoco.mj_forward(env.model, env.data)
    episode = PegEpisode(env)
    for _ in range(1500):
        env.data.mocap_pos[env.fixture_mocap_id, 2] += 0.01 * env.model.opt.timestep
        env.data.mocap_pos[env.table_mocap_id, 2] += 0.01 * env.model.opt.timestep
        controller.step(target)
        reason = episode.step()
        if reason:
            break
    assert reason == "success"


def test_real_jam_triggers_force_stop():
    env = PegInHoleEnv()
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    for _ in range(250):
        controller.step(target)
        env.step()
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    entrance_z = tip[2] - 0.005
    env.data.mocap_pos[env.fixture_mocap_id] = (tip[0] + 0.003, tip[1], entrance_z)
    env.data.mocap_pos[env.table_mocap_id, 2] = entrance_z - HOLE_DEPTH
    mujoco.mj_forward(env.model, env.data)
    episode = PegEpisode(env)
    for _ in range(1500):
        env.data.mocap_pos[env.fixture_mocap_id, 2] += 0.01 * env.model.opt.timestep
        env.data.mocap_pos[env.table_mocap_id, 2] += 0.01 * env.model.opt.timestep
        controller.step(target)
        reason = episode.step()
        if reason:
            break
    assert reason == "force_limit"
    assert episode.peak_force_n > 5.0
