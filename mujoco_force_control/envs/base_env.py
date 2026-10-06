"""Base MuJoCo environment for force control learning."""
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


class ForceControlEnv:
    """Wrapper around MuJoCo for robotic arm force control tasks."""

    def __init__(self, robot="panda", render_mode=None, dt=0.002):
        """
        Args:
            robot: Robot model name ("panda")
            render_mode: None, "human" (live viewer), or "rgb_array"
            dt: Simulation timestep (default: 2ms)
        """
        self.robot = robot
        self.render_mode = render_mode
        self.dt = dt

        # Load model
        model_path = Path(__file__).parent / "assets" / robot / f"{robot}.xml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Viewer (optional)
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Sensor indices
        self._init_sensor_indices()
        self.num_actuators = self.model.nu
        self.num_joints = len(self.model.jnt_range)

    def _init_sensor_indices(self):
        """Cache sensor indices for fast access."""
        self.sensor_indices = {}
        for i, sensor_name in enumerate(self.model.sensor_names):
            self.sensor_indices[sensor_name.decode() if isinstance(sensor_name, bytes) else sensor_name] = i

    def _get_sensor_data(self, sensor_name):
        """Get sensor data by name."""
        if sensor_name not in self.sensor_indices:
            return None
        idx = self.sensor_indices[sensor_name]
        adr = self.model.sensor_adr[idx]
        dim = self.model.sensor_dim[idx]
        return self.data.sensordata[adr:adr+dim]

    def _extract_observation(self):
        """Extract current observation from simulation."""
        # Joint positions and velocities (7-DOF arm)
        joint_pos = self.data.qpos[:7].copy()
        joint_vel = self.data.qvel[:7].copy()

        # Actuator forces (motor current proxy)
        motor_current = self.data.actuator_force[:7].copy()

        # End-effector force (from load cell sensor)
        ee_force = np.zeros(3)
        for i in range(3):
            sensor_data = self._get_sensor_data(f"ee_f{'xyz'[i]}")
            if sensor_data is not None:
                ee_force[i] = sensor_data[0]

        obs = {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'motor_current': motor_current,
            'ee_force': ee_force,
            'time': self.data.time,
        }
        return obs

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset to home configuration (slightly randomized)
        self.data.qpos[:7] = np.array([0, -0.5, 0, -1.5, 0, 1.5, 0.5])
        self.data.qpos[:7] += np.random.randn(7) * 0.1  # Small noise
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self._extract_observation()

    def step(self, action):
        """Execute one simulation step.

        Args:
            action: Joint position targets (7D array)

        Returns:
            obs, reward, terminated, info
        """
        # Convert position targets to torques (handled by PD controller externally)
        # Here we just apply the action directly as torque command
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Sync viewer
        if self.viewer is not None:
            self.viewer.sync()

        obs = self._extract_observation()
        reward = 0.0  # Reward is computed externally during training
        terminated = False  # No early termination
        info = {}

        return obs, reward, terminated, info

    def render_rgb(self, camera_name='arm_camera', height=224, width=224):
        """Render RGB image from specified camera.

        Args:
            camera_name: Name of camera in MJCF (if available)
            height, width: Image resolution

        Returns:
            (height, width, 3) RGB image as uint8
        """
        # Try to find camera by name (fallback to default)
        camera_id = 0
        for i in range(self.model.ncam):
            if self.model.camera_names[i].decode() == camera_name:
                camera_id = i
                break

        # Render
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.enable_depth_rendering()
        mujoco.mj_renderOffscreen(self.model, self.data, renderer._renderer)

        rgb = renderer.render()
        return rgb

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()


if __name__ == "__main__":
    # Test: Load env and run a few steps
    env = ForceControlEnv(robot="panda", render_mode=None)
    obs = env.reset()

    print(f"Joint positions: {obs['joint_pos']}")
    print(f"Joint velocities: {obs['joint_vel']}")
    print(f"Motor currents: {obs['motor_current']}")
    print(f"EE force: {obs['ee_force']}")

    # Run a few steps with zero control
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))

    print(f"After 10 steps - Joint positions: {obs['joint_pos']}")
    print("✓ Environment test passed")
