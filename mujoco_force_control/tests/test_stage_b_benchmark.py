import numpy as np
import pytest
import mujoco

from benchmark_insertion import TrialConfig, ViewerClosed, classify_failure, run_trial
from envs.peg_contact import PegContactForce
from envs.peg_episode import PegEpisode
from envs.peg_in_hole import PegInHoleEnv, WALL_COUNT
from sensors.noisy_wrench import NoisyWrench


def test_sensor_latency_noise_and_seed_are_reproducible():
    kwargs = dict(seed=7, noise_std=(0.1,) * 6, latency_steps=2)
    a = NoisyWrench(**kwargs)
    b = NoisyWrench(**kwargs)
    for step in range(5):
        x = a.update(float(step), np.full(6, step))
        y = b.update(float(step), np.full(6, step))
        if step < 2:
            assert x is None and y is None
        else:
            assert x is not None and y is not None
            assert x[0] == step - 2
            np.testing.assert_array_equal(x[1], y[1])
            assert not np.array_equal(x[1], np.full(6, step - 2))
    with pytest.raises(ValueError, match="timestamps"):
        a.update(4.0, np.zeros(6))


def test_zero_noise_and_latency_returns_exact_current_reading():
    sensor = NoisyWrench(seed=0, noise_std=(0.0,) * 6, latency_steps=0)
    time, reading = sensor.update(1.0, np.arange(6.0))
    assert time == 1.0
    np.testing.assert_array_equal(reading, np.arange(6.0))
    with pytest.raises(ValueError, match="timestamps"):
        sensor.update(1.0, np.zeros(6))


@pytest.mark.parametrize(("kwargs", "geometric", "force"), [
    ({}, "entry_failure", "stalled"),
    ({"lateral_error": 0.001}, "surface_miss", "stalled"),
    ({"surface_contact": False, "radial_distance": 0.019}, "drift_off", "stalled"),
    ({"depth": 0.01}, "unclassified", "stalled"),
    ({"surface_seconds": 0.9}, "entry_failure", "unstable_contact"),
    ({"lateral_exceeded": True}, "entry_failure", "unstable_contact"),
    ({"max_above_s": 1.0, "lateral_exceeded": True}, "entry_failure", "excessive_pressing"),
])
def test_failure_classification_precedence(kwargs, geometric, force):
    settings = dict(depth=0.0, lateral_error=0.0, surface_contact=True, radial_distance=0.0,
                    surface_seconds=1.0, max_above_s=0.0, lateral_exceeded=False)
    settings.update(kwargs)
    assert classify_failure(**settings) == (geometric, force)


def test_paired_trial_reproducibility_and_fixed_fixture():
    config = TrialConfig(seed=4, timeout_s=0.04)
    for method in ("position", "hybrid", "impedance"):
        first = run_trial(method, config, record_trace=True)
        second = run_trial(method, config, record_trace=True)
        assert first == second
        assert first["reason"] == "timeout"
        assert first["fixture_position_m"] == first["final_fixture_position_m"]
        assert len(first["trace"]) == 20
        assert first["trace"][0]["observed_wrench_local"] is None
        assert first["trace"][2]["observation_time_s"] == pytest.approx(first["trace"][0]["raw_wrench_time_s"])
        assert first["force_failure_mode"] == "unstable_contact"


def test_observer_does_not_change_trial_results():
    config = TrialConfig(seed=4, timeout_s=0.04)
    observed_times = []
    with_observer = run_trial("position", config, on_step=lambda env: observed_times.append(float(env.data.time)))
    assert with_observer == run_trial("position", config)
    assert len(observed_times) == 20
    assert observed_times == sorted(observed_times)


def test_viewer_close_aborts_without_a_trial_result():
    def close_viewer(env):
        raise ViewerClosed

    with pytest.raises(ViewerClosed):
        run_trial("position", TrialConfig(timeout_s=0.04), on_step=close_viewer)


def test_noise_changes_hybrid_control_but_not_position_control():
    clean = TrialConfig(seed=2, timeout_s=0.4, noise_force_n=0.0, noise_torque_nm=0.0)
    noisy = TrialConfig(seed=2, timeout_s=0.4, noise_force_n=0.5, noise_torque_nm=0.05)
    clean_hybrid = run_trial("hybrid", clean)
    noisy_hybrid = run_trial("hybrid", noisy)
    assert abs(clean_hybrid["final_depth_m"] - noisy_hybrid["final_depth_m"]) > 1e-6
    clean_position = run_trial("position", clean)
    noisy_position = run_trial("position", noisy)
    assert clean_position["final_depth_m"] == noisy_position["final_depth_m"]


@pytest.mark.parametrize("method", ["position", "hybrid", "impedance"])
def test_robot_driven_insertion(method):
    result = run_trial(method, TrialConfig(seed=0))
    assert result["reason"] == "success"
    assert 0.020 <= result["final_depth_m"] <= 0.025
    assert result["peak_force_n"] <= 5.0
    assert result["final_angle_rad"] <= np.deg2rad(0.2)
    assert result["fixture_position_m"] == result["final_fixture_position_m"]


def test_randomized_faceted_hole_clearance_against_collisions():
    env = PegInHoleEnv()
    episode = PegEpisode(env)
    peg = episode.contact.peg_geom
    walls = {
        env._named_id(mujoco.mjtObj.mjOBJ_GEOM, f"task_hole_wall_{i}")
        for i in range(WALL_COUNT)
    }
    rng = np.random.default_rng(42)
    for _ in range(40):
        direction = rng.uniform(0, 2 * np.pi)
        for radius, expected_seated, expected_contact in (
            (0.00005, True, False), (0.0002, False, True),
        ):
            tip = env.data.site_xpos[env.peg_tip_id].copy()
            env.data.mocap_pos[env.fixture_mocap_id] = tip + [
                radius * np.cos(direction), radius * np.sin(direction), 0.022,
            ]
            mujoco.mj_forward(env.model, env.data)
            wall_contact = any(
                contact.dist < 0 and peg in (contact.geom1, contact.geom2)
                and (int(contact.geom1) in walls or int(contact.geom2) in walls)
                for contact in env.data.contact
            )
            assert episode._seated() is expected_seated
            assert wall_contact is expected_contact


def test_randomized_tilted_clearance_uses_hole_frame():
    env = PegInHoleEnv()
    episode = PegEpisode(env)
    original_quat = env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7].copy()
    rng = np.random.default_rng(23)
    for _ in range(10):
        direction = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(direction), np.sin(direction), 0.0])
        quat = np.empty(4)
        mujoco.mju_axisAngle2Quat(quat, axis, rng.uniform(0, np.deg2rad(0.1)))
        peg_quat = np.empty(4)
        mujoco.mju_mulQuat(peg_quat, quat, original_quat)
        env.data.qpos[env.peg_qpos_address + 3 : env.peg_qpos_address + 7] = peg_quat
        env.data.mocap_quat[env.fixture_mocap_id] = quat
        mujoco.mj_forward(env.model, env.data)
        rotation = env.data.site_xmat[env.hole_site_id].reshape(3, 3)
        tip = env.data.site_xpos[env.peg_tip_id].copy()
        env.data.mocap_pos[env.fixture_mocap_id] = tip - rotation @ np.array([0.00005, 0.0, -0.022])
        mujoco.mj_forward(env.model, env.data)
        assert episode._seated()
        assert PegContactForce(env).measure().contact_count == 0


def test_surface_contact_force_comes_from_peg_fixture_contacts():
    env = PegInHoleEnv()
    tip = env.data.site_xpos[env.peg_tip_id].copy()
    env.data.mocap_pos[env.fixture_mocap_id] = tip + [0.003, 0.0, 0.0005]
    mujoco.mj_forward(env.model, env.data)
    force = PegContactForce(env)
    surface = force.measure(surface_only=True)
    assert surface.contact_count > 0
    assert surface.magnitude_sum > 0
    assert surface.magnitude_sum <= force.measure().magnitude_sum
