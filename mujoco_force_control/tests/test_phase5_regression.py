import numpy as np
import pytest

from controllers.force_controller import EndEffectorForceController
from controllers.pd_position_controller import ARM_JOINTS, PDPositionController
from controllers.phase4_controllers import CascadedForceController
from envs.base_env import BaseEnv
from run_position_demo import run_demo
from sensors import EndEffectorLoadCell
from sensors.torque_observers import (
    ExtendedKalmanTorqueObserver,
    SlidingModeTorqueObserver,
)


@pytest.mark.parametrize(
    ("trajectory", "duration"),
    [("square", 1.0), ("step", 0.0), ("sine", -1.0), ("step", float("nan"))],
)
def test_run_demo_rejects_invalid_arguments(trajectory, duration):
    with pytest.raises(ValueError):
        run_demo(trajectory, duration)


def test_sensor_reads_are_copies_and_reset_restores_keyframe_state():
    env = BaseEnv()
    env.reset("home")
    positions = env.read_sensor("joint1_pos")
    positions[0] = 123.0
    assert env.read_sensor("joint1_pos")[0] != 123.0

    env.step(5)
    env.reset("home")

    assert env.data.time == 0.0
    np.testing.assert_allclose(env.read_sensor("joint1_pos"), env.model.key_qpos[0, :1])


def test_pd_controller_rejects_malformed_and_nonfinite_targets():
    controller = PDPositionController(BaseEnv())
    with pytest.raises(ValueError, match="shape"):
        controller.step(np.zeros(len(ARM_JOINTS) - 1))
    with pytest.raises(ValueError, match="finite"):
        controller.step(np.full(len(ARM_JOINTS), np.inf))


def test_pd_controller_saturates_commands_at_joint_actuator_limits():
    env = BaseEnv()
    controller = PDPositionController(env)

    torques = controller.step(np.full(len(ARM_JOINTS), 1000.0))

    limits = env.model.actuator_forcerange[controller.actuator_ids]
    np.testing.assert_array_less(torques, limits[:, 1] + 1e-12)
    np.testing.assert_array_less(limits[:, 0] - 1e-12, torques)


def test_torque_observer_reset_clears_state_and_reinitializes():
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env)
    ekf = ExtendedKalmanTorqueObserver(controller)
    sliding = SlidingModeTorqueObserver(controller)

    ekf.state[-1] = 2.0
    sliding.estimate[-1] = 2.0
    ekf.reset()
    sliding.reset()

    np.testing.assert_array_equal(ekf.update(), np.zeros(len(ARM_JOINTS)))
    np.testing.assert_array_equal(sliding.update(), np.zeros(len(ARM_JOINTS)))
    np.testing.assert_array_equal(ekf.state[-len(ARM_JOINTS) :], np.zeros(len(ARM_JOINTS)))
    np.testing.assert_array_equal(sliding.estimate, np.zeros(len(ARM_JOINTS)))


def test_load_cell_reports_missing_sensor_by_name():
    env = BaseEnv()

    with pytest.raises(KeyError, match="missing_force"):
        EndEffectorLoadCell(env, force_sensor="missing_force")


def test_force_controller_rejects_malformed_wrench():
    controller = EndEffectorForceController(PDPositionController(BaseEnv()))
    with pytest.raises(ValueError, match="shape"):
        controller.step(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        controller.step(np.full(6, np.nan))


def test_cascaded_force_controller_rejects_nonpositive_timestep():
    cascaded = CascadedForceController(
        EndEffectorForceController(PDPositionController(BaseEnv()))
    )

    with pytest.raises(ValueError, match="dt"):
        cascaded.step(np.zeros(6), dt=0.0)
