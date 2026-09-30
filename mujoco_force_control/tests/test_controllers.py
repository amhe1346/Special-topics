import numpy as np

from controllers.pd_position_controller import ARM_JOINTS, PDPositionController
from envs.base_env import BaseEnv
from run_position_demo import run_demo


def test_pd_controller_reads_home_pose_from_encoder_sensors():
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env)

    positions, velocities = controller.read_joint_state()

    np.testing.assert_allclose(positions, env.model.key_qpos[0, :7])
    np.testing.assert_allclose(velocities, np.zeros(len(ARM_JOINTS)))


def test_pd_step_response_settles_with_bounded_overshoot():
    times, targets, measured, errors = run_demo("step", 3.0)
    post_step = np.flatnonzero(times >= 0.5)
    post_step_errors = np.abs(errors[post_step])
    remaining_max_error = np.maximum.accumulate(post_step_errors[::-1])[::-1]
    settled = np.flatnonzero(remaining_max_error <= 0.01)

    assert settled.size > 0
    assert settled[0] * 0.002 < 1.0
    assert np.max(measured[post_step] - targets[post_step]) < 0.02
    assert abs(errors[-1]) < 0.005


def test_pd_tracks_sinusoidal_joint_target():
    _, _, _, errors = run_demo("sine", 3.0)

    assert np.sqrt(np.mean(errors**2)) < 0.03