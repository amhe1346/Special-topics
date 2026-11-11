import numpy as np
import mujoco

from controllers.force_controller import EndEffectorForceController
from controllers.pd_position_controller import ARM_JOINTS, PDPositionController
from controllers.phase4_controllers import (
    CartesianAdmittanceController,
    CartesianImpedanceController,
    CascadedForceController,
    HybridPositionForceController,
)
from envs.base_env import BaseEnv
from run_position_demo import run_demo
from sensors import EndEffectorLoadCell
from sensors.torque_observers import (
    ExtendedKalmanTorqueObserver,
    SlidingModeTorqueObserver,
)


def test_pd_controller_reads_home_pose_from_encoder_sensors():
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env)

    positions, velocities = controller.read_joint_state()

    np.testing.assert_allclose(positions, env.model.key_qpos[0, :7])
    np.testing.assert_allclose(velocities, np.zeros(len(ARM_JOINTS)))


def test_actuator_force_feedback_reports_joint_torque_and_checks_limits():
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env)
    target = np.array([0.2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    commanded_torques = controller.step(target)
    controller.check_torque_limits(commanded_torques)
    env.step()

    actuator_forces = controller.read_actuator_forces()
    joint_torques = controller.read_joint_torques()
    np.testing.assert_allclose(
        actuator_forces, env.data.actuator_force[controller.actuator_ids]
    )
    np.testing.assert_allclose(
        joint_torques, actuator_forces * controller.actuator_gears
    )
    controller.check_torque_limits()


def test_torque_limit_check_reports_out_of_range_torque():
    env = BaseEnv()
    controller = PDPositionController(env)
    excessive_torques = np.zeros(len(ARM_JOINTS))
    excessive_torques[0] = 88.0

    try:
        controller.check_torque_limits(excessive_torques)
    except RuntimeError as error:
        assert "joint1" in str(error)
        assert "limit exceeded" in str(error)
    else:
        raise AssertionError("an out-of-range torque should raise RuntimeError")


def test_torque_observers_estimate_applied_joint_disturbance():
    env = BaseEnv()
    env.reset("home")
    controller = PDPositionController(env)
    ekf = ExtendedKalmanTorqueObserver(controller)
    sliding_mode = SlidingModeTorqueObserver(controller)
    target, _ = controller.read_joint_state()
    env.data.qfrc_applied[controller.dof_indices[0]] = 5.0

    for _ in range(100):
        controller.step(target)
        env.step()
        ekf_estimate = ekf.update()
        sliding_estimate = sliding_mode.update()

    np.testing.assert_allclose(ekf_estimate[0], 5.0, atol=0.1)
    np.testing.assert_allclose(sliding_estimate[0], 5.0, atol=0.1)
    np.testing.assert_allclose(ekf_estimate[1:], 0.0, atol=0.1)
    np.testing.assert_allclose(sliding_estimate[1:], 0.0, atol=0.1)


def test_end_effector_load_cell_senses_external_force():
    env = BaseEnv()
    env.reset("home")
    load_cell = EndEffectorLoadCell(env)
    baseline_force = load_cell.read_force()
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    env.data.xfrc_applied[hand_id, :3] = [0.0, 0.0, 10.0]
    env.step()
    loaded_force = load_cell.read_force()

    assert baseline_force.shape == (3,)
    assert load_cell.read_torque().shape == (3,)
    assert np.linalg.norm(loaded_force - baseline_force) > 5.0
    np.testing.assert_allclose(load_cell.read_wrench()[:3], loaded_force)


def test_end_effector_force_controller_maps_wrench_to_joint_torques():
    env = BaseEnv()
    env.reset("home")
    joint_controller = PDPositionController(env)
    force_controller = EndEffectorForceController(
        joint_controller, kp=(0.5, 0.5, 0.5, 0.2, 0.2, 0.2)
    )
    measured_wrench = force_controller.read_wrench()
    target_wrench = measured_wrench.copy()
    target_wrench[0] += 1.0
    rotation = env.data.site_xmat[force_controller.site_id].reshape(3, 3)
    jacobian_position = np.zeros((3, env.model.nv))
    jacobian_rotation = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(
        env.model,
        env.data,
        jacobian_position,
        jacobian_rotation,
        force_controller.site_id,
    )
    expected_torques = env.data.qfrc_bias[joint_controller.dof_indices].copy()
    expected_torques += (
        jacobian_position[:, joint_controller.dof_indices].T
        @ (rotation @ np.array([0.5, 0.0, 0.0]))
    )

    applied_torques = force_controller.step(target_wrench)

    np.testing.assert_allclose(applied_torques, expected_torques)
    joint_controller.check_torque_limits(applied_torques)


def test_hybrid_controller_can_disable_force_axes():
    env = BaseEnv()
    env.reset("home")
    joint_controller = PDPositionController(env)
    hybrid = HybridPositionForceController(
        joint_controller, force_axes=(False, False, False, False, False, False)
    )
    target, _ = joint_controller.read_joint_state()
    expected = joint_controller.step(target)
    actual = hybrid.step(target, np.full(6, 100.0))

    np.testing.assert_allclose(actual, expected)


def test_cascaded_force_controller_limits_target_and_resets_integral():
    env = BaseEnv()
    env.reset("home")
    force_controller = EndEffectorForceController(PDPositionController(env))
    cascaded = CascadedForceController(force_controller, wrench_limit=10.0)

    torques = cascaded.step(np.full(6, 1000.0))

    assert np.all(np.abs(cascaded.last_target_wrench) <= 10.0)
    assert np.all(np.isfinite(torques))
    assert np.all(np.abs(cascaded.integral_error) <= cascaded.integral_limit)
    cascaded.reset()
    np.testing.assert_array_equal(cascaded.integral_error, np.zeros(6))


def test_impedance_controller_holds_current_pose_with_gravity_compensation():
    env = BaseEnv()
    env.reset("home")
    joint_controller = PDPositionController(env)
    force_controller = EndEffectorForceController(joint_controller)
    impedance = CartesianImpedanceController(force_controller)
    site_id = force_controller.site_id
    position = env.data.site_xpos[site_id].copy()
    rotation = env.data.site_xmat[site_id].reshape(3, 3).copy()
    expected = env.data.qfrc_bias[joint_controller.dof_indices].copy()

    torques = impedance.step(position, rotation)

    np.testing.assert_allclose(torques, expected, atol=1e-8)
    joint_controller.check_torque_limits(torques)


def test_admittance_controller_integrates_external_wrench_with_bounded_offset():
    env = BaseEnv()
    env.reset("home")
    joint_controller = PDPositionController(env)
    force_controller = EndEffectorForceController(joint_controller)
    impedance = CartesianImpedanceController(force_controller)
    admittance = CartesianAdmittanceController(
        impedance, max_displacement=(0.01, 0.01, 0.01, 0.1, 0.1, 0.1)
    )
    target_wrench = force_controller.read_wrench()
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    env.data.xfrc_applied[hand_id, :3] = [10.0, 0.0, 0.0]

    for _ in range(20):
        env.step()
        torques = admittance.step(target_wrench)

    assert admittance.displacement[0] > 0.0
    assert np.all(np.abs(admittance.displacement) <= admittance.max_displacement)
    assert np.all(np.isfinite(torques))
    admittance.reset()
    np.testing.assert_array_equal(admittance.displacement, np.zeros(6))


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