# MuJoCo Closed-Loop Position/Force Control

A staged learning project for position and force control of a Franka Emika Panda
arm in MuJoCo. The simulation uses MuJoCo's native sensors as stand-ins for
encoders, an end-effector load cell, and motor-current sensing.

## Setup

From the workspace root, enter the project directory, then create and activate
a virtual environment and install the dependencies:

```bash
cd mujoco_force_control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the Phase 0 smoke test and launch the model:

```bash
python -m pytest
python -m envs.base_env
python -m envs.base_env --viewer
```

## Phase 1: Position Control

Run the step or sinusoidal joint-1 target and plot encoder tracking:

```bash
python -m run_position_demo --trajectory step
python -m run_position_demo --trajectory sine --kp 400 --kd 45
```

The controller reads `jointpos`/`jointvel` sensor values and applies saturated
PD torque commands with gravity compensation. The seven arm position servos in
the vendored Panda model are configured as bounded torque motors for this
phase; the gripper actuator remains unchanged. Adjust `--kp` and `--kd` to tune
the response.

## Phase 2: Actuator Force Feedback

The arm model exposes an `actuatorfrc` sensor for each torque motor. Use the
controller to read measured actuator forces or their corresponding joint
torques, and check measured or commanded torques against the configured
actuator force limits:

```python
from controllers.pd_position_controller import PDPositionController
from envs.base_env import BaseEnv

env = BaseEnv()
controller = PDPositionController(env)
target, _ = controller.read_joint_state()
commanded_torques = controller.step(target)
controller.check_torque_limits(commanded_torques)

env.step()
measured_torques = controller.read_joint_torques()
controller.check_torque_limits()
```

Calling `check_torque_limits()` without an argument checks the latest measured
torques. Call it after advancing the simulation to validate the force feedback.

## Phase 2.5: Torque Observers

The extended Kalman filter (EKF) predicts joint motion using the MuJoCo mass
matrix and bias forces, then corrects its position and velocity estimates from
the encoder sensors. The sliding-mode observer estimates disturbance torque
from the inverse-dynamics residual and measured velocity changes. Both return
joint-space disturbance torque estimates:

```python
from sensors import ExtendedKalmanTorqueObserver, SlidingModeTorqueObserver

ekf = ExtendedKalmanTorqueObserver(controller)
sliding_mode = SlidingModeTorqueObserver(controller)

env.step()
ekf_disturbance = ekf.update()
sliding_disturbance = sliding_mode.update()
```

Call each observer once per simulation step. Estimates represent unmodeled or
external joint torques, including contact effects; model mismatch and velocity
measurement noise can affect them.

## Phase 3: End-Effector Force Control

A six-axis load cell at the Panda hand reports force (N) and moment (N m) in
its local frame. `EndEffectorForceController` compares that wrench with a
six-component target, maps the proportional correction through the
end-effector Jacobian transpose, adds gravity compensation, and applies
actuator limits:

```python
import numpy as np

from controllers.force_controller import EndEffectorForceController
from controllers.pd_position_controller import PDPositionController
from envs.base_env import BaseEnv
from sensors import EndEffectorLoadCell

env = BaseEnv()
env.reset("home")
joint_controller = PDPositionController(env)
load_cell = EndEffectorLoadCell(env)
force_controller = EndEffectorForceController(joint_controller, load_cell)
target_wrench = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])

for _ in range(1000):
    force_controller.step(target_wrench)
    env.step()
    measured_wrench = force_controller.read_wrench()
```

Wrench targets use the load-cell frame and are ordered as three force
components followed by three moments. Force regulation requires contact with an
environment; in free space the arm cannot sustain a nonzero force setpoint.
Tune the six proportional gains for the contact stiffness and simulation step.

## Phase 4: Hybrid and Compliant Control

`HybridPositionForceController` combines joint-space position tracking with
Cartesian force feedback on selected load-cell axes. Its six-element
`force_axes` mask is ordered as force XYZ then moment XYZ; the default controls
force along the sensor's Z axis while tracking the requested joint positions:

```python
from controllers import HybridPositionForceController

hybrid = HybridPositionForceController(joint_controller, force_controller)
target_joints, _ = joint_controller.read_joint_state()
target_wrench = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])

for _ in range(1000):
    hybrid.step(target_joints, target_wrench)
    env.step()
```

`CascadedForceController` uses an outer PI loop to adjust the desired wrench
and the Phase 3 wrench-feedback controller as its inner loop. Wrench and
integral limits are configurable:

```python
from controllers import CascadedForceController

cascaded = CascadedForceController(force_controller)
for _ in range(1000):
    cascaded.step(target_wrench)
    env.step()
```

`CartesianImpedanceController` tracks a world-frame target position and
rotation with a virtual spring-damper. `CartesianAdmittanceController` uses
the measured wrench error and virtual mass-damper-spring dynamics to adjust a
bounded pose offset, then tracks that pose with the impedance controller:

```python
from controllers import (
    CartesianAdmittanceController,
    CartesianImpedanceController,
)

impedance = CartesianImpedanceController(force_controller)
site_id = force_controller.site_id
target_position = env.data.site_xpos[site_id].copy()
target_rotation = env.data.site_xmat[site_id].reshape(3, 3).copy()
impedance.step(target_position, target_rotation)

admittance = CartesianAdmittanceController(impedance)
for _ in range(1000):
    admittance.step(target_wrench)
    env.step()
```

Call the active controller once per simulation step. These methods are
simulation examples, not safety-rated robot controllers; gain, saturation, and
contact behavior should be validated for each model and task.

## Phase 5: Regression Tests and Project Structure

Run the regression suite and model smoke test from this directory:

```bash
python -m pytest -q
python -m envs.base_env
```

The tests cover model loading, named sensor reads, keyframe reset, joint and
actuator torque feedback, observer response to an applied disturbance,
end-effector wrench sensing, torque mapping, force-axis selection, cascaded
force limits, impedance/admittance behavior, and position-trajectory tracking.
Controller inputs are checked for invalid shapes and non-finite values, and
joint commands are verified against the model's actuator limits.

The project is organized by responsibility: `envs/` contains the MuJoCo
wrapper and Panda MJCF/assets, `controllers/` contains joint and Cartesian
controllers, `sensors/` contains load-cell access and torque observers, and
`tests/` contains the regression suite. `run_position_demo.py` is the
command-line position-tracking example.

## Stage B: Peg-in-Hole Simulation

`PegInHoleEnv` builds a separate task scene from the existing Panda model; the
original demos and model remain unchanged. The scene has an 8 mm diameter peg
with a free joint, a nominal 8.16 mm diameter hole, a collidable table, and two
fixed workspace cameras named `task_left_camera` and `task_right_camera`.

```python
from envs.peg_in_hole import PegInHoleEnv

env = PegInHoleEnv()
env.reset(seed=42, lateral_range=0.003)
measurement = env.measure_geometry()
print(measurement)
```

Reset places the peg between the nearly closed fingers, commands the gripper
closed, puts the peg tip 15 mm above the hole entrance, and samples independent
horizontal fixture offsets uniformly within `[-lateral_range, lateral_range]`
in metres. The peg is held by simulated finger contacts rather than a weld.
The gripper actuator uses the Panda model's 0–255 control range: setting
`env.data.ctrl[env.gripper_actuator_id] = 255` opens it and allows the peg to
fall. Apply an arm controller during simulation steps to hold or move the arm;
reset positions the peg but does not execute a pick-up trajectory. The hole
entrance's local Z axis points outward; insertion depth
is positive inward. `measure_geometry()` reports true simulator depth, the
peg-axis intersection's lateral distance from the entrance center, and angular
misalignment. Its `geometric_candidate` checks an entrance-plane intersection,
20–25 mm depth, and at most 0.08 mm lateral error, with a 1e-12 m numerical
roundoff allowance. This measurement is not itself a full success decision.

The simulation-only `PegEpisode` evaluator advances the environment one
physics step at a time and returns a termination reason:

```python
from controllers.pd_position_controller import PDPositionController
from envs.peg_episode import PegEpisode
from envs.peg_in_hole import PegInHoleEnv

env = PegInHoleEnv()
env.reset(seed=42)
controller = PDPositionController(env)
target, _ = controller.read_joint_state()
episode = PegEpisode(env)
while episode.reason is None:
    controller.step(target)  # replace with the desired arm command
    episode.step()           # do not also call env.step()
print(episode.reason, episode.peak_force_n, episode.above_reference_intervals)
```

Construct a new episode after each reset. Success requires actual crossing of
the entrance plane, 20-25 mm depth, at most 0.08 mm axis offset at the
entrance, tilt at most 0.2 degrees, and 0.1 seconds of continuously valid
geometry. To account for tilt and finite peg length, the largest axis offset at
the entrance and tip plus the conservative projected radius
`peg_radius / cos(tilt)` must fit within the nominal hole radius, with a
1e-12 m geometric roundoff allowance. A failed geometry check resets the
dwell. These are geometric simulator ground truth checks, not a physical
measurement or damage test.

Defaults in `EpisodeConfig` are provisional: 1 N analysis reference, 5 N
hard stop, 20 s timeout, and a workspace centered at the initial hole entrance
with +/-0.3 m in X/Y and -0.05 to +0.3 m in Z. The drop heuristic flags
either >20 mm peg displacement relative to the initial hand-frame position or
0.1 s without contact with either finger. Call
`episode.set_release_intent(True)` **before** an intentional release; opening
the gripper alone does not set intent. Release intent suppresses drop
detection until explicitly cleared, so it must only be set for a deliberate
release. These heuristics and workspace bounds need task-specific validation.

Every step checks for invalid simulation state/warnings, then force limit,
workspace, dropped peg, success, and timeout in that precedence order.
Termination prevents further `PegEpisode.step()` calls; it does not implement a
hardware emergency stop or clear motor commands. The force limit is checked
after a physics step, so the recorded peak can exceed it. Peg contact force is
the **sum of magnitudes** of MuJoCo contact forces against the hole walls and
table, including friction; finger contacts are excluded. This differs from
the net world-frame force vector also recorded per sample and from the wrist
load-cell wrench. Peak force, samples, and continuous intervals strictly above
the reference are retained through success or failure; interval edges have
one-physics-step resolution. The caller must not step the environment outside
the evaluator during an episode.

Force thresholds are exploratory simulation values, not real-world limits.
The traces in `results/force_probes/` were generated with:

```sh
python3 run_contact_probe.py --speed 0.01 > results/force_probes/speed_0.01.csv
python3 run_contact_probe.py --speed 0.02 > results/force_probes/speed_0.02.csv
```

Each probe settles a PD-held peg, then raises the kinematic fixture and table
at the specified speed. The aligned and pressing cases use zero initial
lateral offset; the jam uses 3 mm. At 0.01/0.02 m/s the measured peak forces
were 0.287/0.000 N aligned, 10.645/15.583 N jammed, and 5.720/6.268 N
pressing, respectively. The 1 N reference and 5 N stop are initial choices
separating these particular trajectories, not a statistical validation.
These calibration probes deliberately continue past normal episode success
and force-stop conditions to observe pressing; the slower aligned probe also
reaches the bottom. They do not model a robot-driven approach, deformable
parts, sensor noise, or actual hardware safety. CSV rows record scenario,
commanded speed/offset, time, depth, contact magnitude, and world force vector.

The hole uses 64 tangent box walls as a faceted approximation, and the contact,
friction, grasp and table settings are provisional. The table surface is at the
bottom of the hole, so the peg can land on it after release. The fixture and
table are kinematic mocap bodies, not dynamic or deformable parts. Reset starts
with the peg between the fingers; a pick-up policy and validated grasp success
detection are not implemented. The cameras can be rendered with
MuJoCo's renderer, subject to the system's rendering backend.

### Robot-driven baseline trials

The Stage B benchmark starts with an **already grasped peg**, holds the fixture
fixed, and moves the robot. It checks both fingers are touching after a 0.5 s
settling period; an invalid initial grasp aborts the run explicitly. From the
project directory:

```sh
# Three paired seeds per controller; results are JSON lines on stdout.
python3 benchmark_insertion.py --trials 3

# Watch one robot-driven impedance trial in the interactive MuJoCo viewer.
python3 benchmark_insertion.py --viewer --method impedance --seed 0 --trials 1

# Save per-step traces to a new directory and summaries separately.
python3 benchmark_insertion.py --method all --trials 3 \
  --trace-dir results/my-traces > results/my-summary.jsonl
```

The viewer requires a graphical desktop. It runs the actual benchmark at
approximately real-time speed, initially zoomed toward the fixture. The peg is
small: use the mouse to adjust the view. After the outcome is printed to the
terminal, close the window to finish or continue to the next trial. Closing it
before termination aborts the trial and exits with an error instead of
recording a completed outcome. Do not interact with or perturb the robot if
you want reproducible benchmark measurements.

The trace directory must not already exist. The shell can overwrite the
summary file, so choose a new name. `--help` lists parameters. Each method
uses the same pose for a given seed: independent uniform hole offsets within
+/-0.05 mm in X/Y and fixture tilt components within +/-0.03 degrees around
world X/Y. The table stays horizontal; the fixture tilt is provisional. The
default downward command is 0.005 m/s and the episode timeout is 12 s. The
fixture center and tilt, config, MuJoCo version, terminal reason, final pose
errors, contact peak, above-reference intervals and failure labels are in each
summary. Full traces include time-stamped raw and delayed noisy local-frame
wrenches, peg/environment contact forces, top-surface contact and seating.

The baselines are:

- `position`: damped-least-squares Cartesian position/rotation tracking
  translated into joint-PD targets.
- `hybrid`: the same position target plus the existing axial load-cell force
  feedback controller, with a zero-tared local-Z wrench target.
- `impedance`: the existing Cartesian spring-damper controller.

These are initial controllers, not extensively tuned or proven optimal. They use simulator
ground truth to set the initial lateral target and orientation; camera-based
pose estimation is not implemented. The load cell is tared once after
settling. Its raw reading still contains dynamics and changing gravity loads,
so it must not be interpreted as the isolated peg/hole force. The hybrid
controller uses the noisy delayed reading; the other two do not use force
feedback. The ideal contact forces and geometry remain independent evaluator
signals for all controllers, including the safety stop.

The provisional force-sensor model adds independent Gaussian noise to each
local wrench component (standard deviations 0.05 N and 0.005 N m) and delays
it by two physics steps (4 ms at the default timestep). The noise stream is
seeded separately from fixture randomization. Before the buffer fills, the
hybrid controller uses position control only; trace observations are null.
Other sensors, including cameras and joint encoders, currently remain ideal.

Failure labels use a provisional simulator-only surface-contact heuristic:
peg/fixture or peg/table contacts whose contact point is within 1 mm of the
hole entrance plane. Contact duration is cumulative; lateral and axial force
are the net contact force in the hole frame. Force labels use the precedence
excessive pressing (surface force strictly above 1 N continuously for at least
1 s), then unstable contact (under 1 s cumulative surface contact or lateral
force greater than axial), then stalled. The scalar surface force is the sum
of individual contact magnitudes, not the net vector magnitude. Geometric
labels require final surface contact and tip depth <= 1 mm for entry failure
(aligned) or surface miss (not aligned); tip distance over 18 mm from the
fixture center is drift-off. Other cases, including partial insertions, are
`unclassified` rather than forced into a category. These heuristic labels
need validation before use in comparative plots.

This is a simulation benchmark prototype, not a validated real-world safety
system. Camera/encoder noise, a pickup policy, broader controller tuning,
and statistically powered performance evaluation remain future work.

The initial joint-target update gain of 0.05 was too aggressive: the 10 paired
seeds 0-9 in `results/stage_b_pilot.jsonl` produced no position or hybrid
successes. After reducing the gain to 0.005 using early pilot trials, the
separate seeds 10-19 in `results/stage_b_evaluation.jsonl` produced:

| Controller | Success | Force stop | Timeout |
|---|---:|---:|---:|
| Position | 8 | 1 | 1 |
| Hybrid | 10 | 0 | 0 |
| Impedance | 9 | 0 | 1 |

These are only ten paired trials in a narrow simulator distribution and are
not evidence of comparative superiority. One position trial reached 44.9 N
before its 5 N stop was detected after a physics step. The hard stop is
therefore **not a force cap**, and the simulated peak and threshold must not
be used as hardware-safety claims.

A separate five-seed stress run with +/-0.5 mm fixture offsets and +/-0.1
degree tilt components is saved in `results/stage_b_stress.jsonl`: position
had four successes and one force stop; hybrid and impedance each had five
successes. These trials also use exact simulated fixture and peg positions
for initial alignment and are not a vision-based robustness evaluation.

## Stage C: pi0.5-DROID integration and pilot data

The adapter targets the official OpenPI `pi05_droid` configuration and
`gs://openpi-assets/checkpoints/pi05_droid` checkpoint. The published DROID
example requires one external RGB camera, a wrist RGB camera, seven joint
positions, a normalized gripper position and a language prompt. A
hand-mounted `task_wrist_camera` has been added to the task scene; choose
either fixed workspace camera for the external view. Camera placement is
provisional and not calibrated to the DROID training setup.

Run a **mock-policy**, non-actuating integration check:

```sh
MUJOCO_GL=egl python3 run_pi05_preview.py --policy mock --camera left
```

The mock returns zero actions and is **not pi0.5**. This command renders the
two camera views and prints the first action preview; it does not step the
robot or execute actions. On systems with a working graphical backend, omit
`MUJOCO_GL=egl` if necessary. An isolated OpenPI installation and checkpoint
cache can be kept in the ignored `.openpi/` directory. When that setup exists,
start the localhost-only server in one terminal:

```sh
cd mujoco_force_control
OPENPI_DATA_HOME="$PWD/.openpi/model-cache" XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  .openpi/.venv/bin/python run_openpi_local_server.py --port 8000
```

Once it reports that it is listening, run the preview in another terminal:

```sh
cd mujoco_force_control
MUJOCO_GL=egl .openpi/client-venv/bin/python run_pi05_preview.py --policy local-server --port 8000
```

Stop the server with Ctrl+C when finished. This wrapper and client use only
`127.0.0.1`; the official unmodified server binds to all interfaces by default.
Confirm the server configuration yourself: client metadata does not verify its
checkpoint and the output explicitly reports that limitation. The separate
client environment avoids downgrading the project's NumPy 2 to the official
client's NumPy <2 requirement.

On this machine, the pinned OpenPI revision and `pi05_droid` checkpoint were
loaded in an isolated Python 3.11 server and a non-actuating local inference
completed. The downloaded checkpoint cache is approximately 12 GB. For a fresh
setup, clone OpenPI with submodules into `.openpi`, check out the pinned
revision below, run `uv sync --frozen --python 3.11` there, and create
`.openpi/client-venv` with MuJoCo, Pillow and the official `openpi-client`.
The optional server/client are not installed by this project's requirements.
Keep `.openpi` private (`chmod 700 .openpi`): OpenPI's cache helper makes
cache directories world-accessible, so the private parent protects the local
checkpoint from other users on a shared machine.

Each observation uses uint8 RGB resized with aspect-ratio-preserving padding
to 224x224, a synchronized simulator timestamp, seven joint angles in radians,
and gripper closed fraction `1 - width / 0.08` (0 open, 1 closed). The
external camera name's `left` suffix in the OpenPI key refers to the left
stereo image, not which workspace camera was chosen. The adapter sends raw
state; checkpoint normalization belongs to OpenPI. No force observation is
added to the fixed backbone.

The official example describes a 10x8 chunk, but the pinned `pi05_droid`
configuration specifies a 15-step horizon, and actual checkpoint inference
returned a 15x8 chunk. It contains seven normalized joint-velocity commands
and a gripper-position command at 15 Hz. The adapter validates the
shape and finite values, clips to [-1, 1], binarizes the gripper at >0.5, and
reports the DROID implementation's 0.2-radian-per-command joint delta as a
**preview only** by the preview command. The separate simulation rollout
below applies more restrictive provisional targets; neither is a validated
hardware controller.

### Simulation rollout and pilot collection

With the localhost server running, execute a simulation-only trial. Output
directories must not already exist:

```sh
MUJOCO_GL=egl .openpi/client-venv/bin/python run_pi05_rollout.py \
  --seed 1 --timeout-s 4 --allow-contact --output results/my_pi05_trial
```

Omit `--allow-contact` to stop after detecting the first peg/fixture or
peg/table contact. This detection and the episode's provisional 5 N force
stop happen **after** a physics step: they cannot prevent contact or force
overshoot. The rollout starts with a grasped peg, replans using only the first
action in each chunk, clamps the target to joint limits and limits the
per-command target offset to 0.1 rad/s divided by 15 Hz. This is **not** an
actual joint-speed limit. An existing saturated PD torque controller executes
the targets; actual speed and measured torque are not separately monitored
for termination. The grasp check requires contact with at least one finger,
not proof of a stable two-finger grasp.

The simulation is frozen while synchronous inference runs. The 15 Hz cadence
is simulation time, not a real-time guarantee; the client bounds its response
wait, but this is not a preemptive hardware watchdog. The rollout also stops
for release requests, stale simulation time, episode failure or timeout, and
clears arm controls on exit, including exceptions. These are development
guards, not safety certification. Never use this loop to operate a robot.

Collect a small set of classical impedance trials separately:

```sh
MUJOCO_GL=egl python3 collect_stage_c.py --output results/my_classical_pilot
```

The default seeds are train 0, 1, 2; validation 10; and test 11. Splits are
disjoint by seed. The manifest contains each outcome; `status: in_progress`
indicates an interrupted or failed collection, and the command reports the
error. Successful completion writes training-only joint-position mean and
standard deviation. Other fields are not normalized, and no standard-deviation
floor or normalization transform is provided.

The per-trial NPZ samples are approximately 10 Hz post-step captures: two RGB
views, joint positions and velocities, gripper fraction, raw local wrist
wrench, peg/fixture contact force, actuator controls, geometry and a
depth-based phase label. Each control drove the immediately preceding physics
step; intermediate controls are not recorded. The wrist wrench includes
gravity and inertia, not just contact. The phase is a heuristic, and the
outcome is per trial. These classical actuator controls are **not** DROID
normalized velocity actions and are not directly suitable as pi0.5 training
targets.

Recorded pi0.5 trials additionally save `observations.npz` with the exact
preprocessed policy-input images and state at inference times. The manifest
records first-action clipped normalized velocity, binary gripper decision,
joint target where applicable, observation time, inference duration and
outcome. Later actions in each model chunk are not logged or executed.
Observations may outnumber decisions if inference times out or fails. The
10 Hz post-step samples and inference observations have different cadences;
join by simulation timestamp, not array index. Exceptions produce an error
manifest and are re-raised; a process interruption can leave an in-progress
manifest or partial files.

One local development collection (`results/stage_c_development_v3`) produced
three training successes, one validation timeout and one test success; it has
277 training samples and is far too small for a meaningful performance claim.
In a separately recorded pi0.5 contact trial (`results/stage_c_pi05_recorded`),
seed 1 ended in a dropped peg at 3.806 s simulation time, with 2.208 N peak
peg contact and 58 inference calls of about 79–93 ms each. It was not a
successful insertion. A separate seed 0 no-contact trial stopped on contact
at 0.607 N after 1.282 s; it was recorded before inference-observation
logging was added. These examples are developmental, not performance or
real-time validation. Repeated trials, a substantially larger representative
dataset, explicit action conversion and normalization, checkpoint-weights
license verification, and deployment-specific safety validation remain
before Stage C can be considered finished for training or real-world use.

Verification references, pinned to OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`:

- [Official DROID checkpoint and serving instructions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README.md)
- [DROID input/output transformation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)
- [DROID inference and action example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)
- [DROID gripper conventions](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py) and [joint-action scaling](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py)
- [OpenPI requirements and checkpoint list](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md), [code license](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE), and [custom DROID fine-tuning guide](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README_train.md)

The OpenPI repository code is Apache-2.0 licensed; independent terms for the
weights have not been verified. OpenPI lists an NVIDIA GPU with over 8 GB
memory for inference; availability and actual memory use must be checked on
the machine running the server.

The Panda model and its assets are vendored from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).
The upstream license is retained in `envs/assets/panda/LICENSE`.

## Stages

- **Phase 0:** Panda model, simulation wrapper, and smoke test.
- **Phase 1:** encoder-like joint sensors and joint-space PD/PID control.
- **Phase 2:** actuator force feedback and torque checks.
- **Phase 2.5:** EKF and sliding-mode torque observers.
- **Phase 3:** end-effector load-cell sensing and force control.
- **Phase 4:** hybrid position/force, cascaded force, and impedance/admittance.
- **Phase 5:** regression tests and project documentation.
- **Phase 6:** simulated CAN bus and virtual MCU nodes.
- **Phase 7:** reproducible controller-paradigm benchmarks.

The environment wrapper resolves sensor data by the sensor names in the MJCF.
Sensors will be added to the Panda model as the relevant project stages are
implemented; requesting an undefined sensor raises `KeyError`.