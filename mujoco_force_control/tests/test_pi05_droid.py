import mujoco
import numpy as np
import pytest
import threading

from envs.peg_in_hole import PegInHoleEnv
from integrations.pi05_droid import (
    CHUNK_SHAPE,
    DROID_MAX_JOINT_DELTA_RAD,
    DroidObservation,
    droid_joint_velocity_to_delta,
    infer_preview,
    preview_actions,
)
from integrations.sim_observation import capture_observation
from run_pi05_preview import LocalOpenPiPolicy


def observation(**changes):
    kwargs = dict(
        time_s=1.0, exterior_rgb=np.full((100, 200, 3), 255, dtype=np.uint8),
        wrist_rgb=np.full((224, 224, 3), 7, dtype=np.uint8),
        joint_position_rad=np.arange(7.0), gripper_closed_fraction=0.25, prompt="Insert the peg",
    )
    kwargs.update(changes)
    return DroidObservation(**kwargs)


def test_observation_contract_and_padding():
    request = observation().request()
    assert set(request) == {
        "observation/exterior_image_1_left", "observation/wrist_image_left",
        "observation/joint_position", "observation/gripper_position", "prompt",
    }
    image = request["observation/exterior_image_1_left"]
    assert image.shape == (224, 224, 3) and image.dtype == np.uint8
    assert np.all(image[:56] == 0) and np.all(image[56:168] == 255) and np.all(image[168:] == 0)
    assert request["observation/wrist_image_left"][0, 0, 0] == 7
    np.testing.assert_array_equal(request["observation/joint_position"], np.arange(7.0))
    np.testing.assert_array_equal(request["observation/gripper_position"], [0.25])


@pytest.mark.parametrize("changes", [
    {"time_s": float("nan")}, {"prompt": "  "}, {"gripper_closed_fraction": 1.1},
    {"joint_position_rad": np.zeros(8)}, {"wrist_rgb": np.zeros((2, 2, 3))},
    {"exterior_rgb": np.zeros((2, 2), dtype=np.uint8)},
])
def test_invalid_observations_raise(changes):
    with pytest.raises(ValueError):
        observation(**changes).request()


def test_action_chunk_is_clipped_binary_and_never_executed():
    class Fake:
        def infer(self, request):
            assert request["prompt"] == "Insert the peg"
            action = np.zeros(CHUNK_SHAPE)
            action[0, :7] = [2, -2, 0.5, 0, 0, 0, 0]
            action[0, 7] = 0.51
            action[1, 7] = 0.5
            return {"actions": action}

    result = infer_preview(Fake(), observation())
    assert len(result) == 15
    np.testing.assert_allclose(result[0].joint_delta_rad, [0.2, -0.2, 0.1, 0, 0, 0, 0])
    assert result[0].gripper_closed and not result[1].gripper_closed


def test_droid_joint_velocity_round_trips_to_per_command_joint_delta():
    action = np.array([0.03, -0.02, 0.0, 0.01, -0.033, 0.0, 0.005])
    delta = droid_joint_velocity_to_delta(action)
    np.testing.assert_allclose(delta, action * DROID_MAX_JOINT_DELTA_RAD)
    np.testing.assert_allclose(delta / DROID_MAX_JOINT_DELTA_RAD, action)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        droid_joint_velocity_to_delta(np.full(7, 1.01))
    with pytest.raises(ValueError, match="final dimension of 7"):
        droid_joint_velocity_to_delta(np.zeros(8))


@pytest.mark.parametrize("response", [{}, {"actions": np.zeros((1, 8))}, {"actions": np.zeros((10, 8))}, {"actions": np.full(CHUNK_SHAPE, np.nan)}])
def test_invalid_policy_output_is_rejected(response):
    with pytest.raises(ValueError):
        preview_actions(response)


def test_wrist_camera_attached_to_hand_and_gripper_state_from_actual_width():
    env = PegInHoleEnv()
    camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "task_wrist_camera")
    assert env.model.cam_bodyid[camera] == env.hand_id
    camera_forward = -env.data.cam_xmat[camera].reshape(3, 3)[:, 2]
    target_direction = env.data.site_xpos[env.peg_tip_id] - env.data.cam_xpos[camera]
    target_direction /= np.linalg.norm(target_direction)
    assert np.dot(camera_forward, target_direction) > 0.99

    class FakeRenderer:
        def update_scene(self, data, camera):
            self.camera = camera

        def render(self):
            return np.full((224, 224, 3), 1 if self.camera == "task_wrist_camera" else 2, dtype=np.uint8)

    before = env.data.qpos.copy()
    sample = capture_observation(env, FakeRenderer(), prompt="Insert")
    np.testing.assert_array_equal(env.data.qpos, before)
    assert sample.time_s == 0
    assert sample.gripper_closed_fraction == pytest.approx(0.925)
    assert sample.wrist_rgb[0, 0, 0] == 1
    assert sample.exterior_rgb[0, 0, 0] == 2
    with pytest.raises(ValueError, match="exterior_camera"):
        capture_observation(env, FakeRenderer(), prompt="Insert", exterior_camera="task_wrist_camera")


def test_gripper_observation_uses_closed_fraction_not_open_fraction():
    env = PegInHoleEnv()
    env.data.qpos[env.model.jnt_qposadr[env.finger_joint_ids]] = 0.04
    mujoco.mj_forward(env.model, env.data)

    class FakeRenderer:
        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros((224, 224, 3), dtype=np.uint8)

    opened = capture_observation(env, FakeRenderer(), prompt="Insert")
    assert opened.gripper_closed_fraction == pytest.approx(0.0)
    env.data.qpos[env.model.jnt_qposadr[env.finger_joint_ids]] = 0.0
    mujoco.mj_forward(env.model, env.data)
    closed = capture_observation(env, FakeRenderer(), prompt="Insert")
    assert closed.gripper_closed_fraction == pytest.approx(1.0)


def test_local_openpi_wire_protocol():
    msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")
    server_module = pytest.importorskip("websockets.sync.server")
    captured = []

    def handler(connection):
        connection.send(msgpack_numpy.packb({"test": True}))
        captured.append(msgpack_numpy.unpackb(connection.recv()))
        actions = np.zeros(CHUNK_SHAPE)
        actions[0, 0] = 0.5
        connection.send(msgpack_numpy.packb({"actions": actions}))

    with server_module.serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            policy = LocalOpenPiPolicy(server.socket.getsockname()[1])
            try:
                actions = infer_preview(policy, observation())
            finally:
                policy.close()
        finally:
            server.shutdown()
            thread.join()
    assert actions[0].joint_delta_rad[0] == pytest.approx(0.1)
    assert len(captured) == 1
    assert captured[0]["observation/wrist_image_left"].shape == (224, 224, 3)
