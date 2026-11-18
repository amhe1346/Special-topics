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

## Stage A: Simulation Protocol (Approved Scope)

Stage A is approved **only for MuJoCo simulation**. There is no real robot in
this study, and this protocol does not authorize physical contact or establish
hardware safety limits. A future hardware study needs its own equipment
specification, calibrated measurements, numeric safety limits, and safety
approval.

The task model uses an 8 mm diameter, 60 mm long peg and a nominal 8.16 mm
diameter, 30 mm deep hole. The 64-wall hole is a faceted approximation.
Scored success requires actual entrance-plane crossing, 20-25 mm insertion
depth inclusive, at most 0.08 mm peg-axis offset at the entrance, at most
0.2 degrees angular error, 0.1 seconds of continuously valid geometry, and
the evaluator's clearance check along the inserted peg. Geometry is measured
from MuJoCo ground truth in the hole frame; these measurements are exact only
with respect to the simulated model.

For paired `position`, `hybrid`, and `impedance` trials, use the same
non-negative integer seed and task configuration across methods. Fixture X/Y
offsets are independently sampled from +/-0.05 mm, and tilt components about
world X/Y from +/-0.03 degrees; the table remains horizontal and lighting is
fixed. Trials start with an already-grasped peg, settle for 0.5 seconds, and
require both finger contacts. The scored episode then has a 12-second timeout
and a nominal 0.005 m/s downward command. Use separate seeds for tuning and
held-out evaluation, and report the exact seed list.

The simulation evaluator uses a 1 N analysis reference and a 5 N
peg/environment contact-force termination threshold, plus workspace bounds of
+/-0.3 m in X/Y and -0.05 to +0.3 m in Z relative to the hole entrance. The
force threshold is checked after a physics step and may overshoot; it is not a
force cap. The commanded approach rate is not an enforced speed limit, and
actuator saturation is not an independent joint-torque safety system. Report
all overshoots and unfavorable outcomes. Analyze force using the sum of
peg/hole and peg/table contact magnitudes, excluding finger contacts; keep
this separate from the hand load-cell wrench and any net force vector. Fref
intervals use samples strictly above 1 N at physics-step resolution.

Record every started trial, including force stops, drops, timeouts, invalid
simulation states, and failed initial grasps (the latter are invalid starts,
not silently discarded). Save data and provenance to a new run directory
under `results/`; do not overwrite existing runs. Keep camera/state streams
aligned by simulation timestamp rather than array index. Force can be retained
for analysis in every ablation but must not enter policy observations in
force-observation-off variants.

## Stage B: Peg-in-Hole Simulation (Complete, Simulation Benchmark)

Stage B is complete for the simulation-only scope. Automated tests cover
seeded reset, geometry thresholds and collision clearance, evaluator success
and failure paths, force accounting, and reproducible sensor noise/latency.
They also execute each classical baseline. Run the focused Stage B suite from
this directory:

```sh
python3 -m pytest -q tests/test_peg_in_hole.py tests/test_peg_episode.py tests/test_stage_b_benchmark.py
```

This completion means the benchmark is runnable and regression-tested; it does
not mean the contact model, heuristic failure labels, force stop, or controller
comparison is validated for hardware or statistically conclusive. The 5 N
threshold is checked after a physics step and has produced a 44.9 N peak in a
prior trial. Stage C's action-conversion and data-quality gates still apply
before policy training.

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
force-stop threshold, 20 s timeout, and a workspace centered at the initial
hole entrance with +/-0.3 m in X/Y and -0.05 to +0.3 m in Z. The robot-driven
benchmark overrides the evaluator timeout to 12 s. The drop heuristic flags
either >20 mm peg displacement relative to the initial hand-frame position or
0.1 s without contact with either finger. Call
`episode.set_release_intent(True)` **before** an intentional release; opening
the gripper alone does not set intent. Release intent suppresses drop
detection until explicitly cleared, so it must only be set for a deliberate
release. These simulation-only heuristic thresholds have regression coverage,
but are not evidence of physical safety or broadly validated task limits.

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
friction, grasp and table settings remain provisional. The table surface is at the
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
before its 5 N force-stop threshold was detected after a physics step. This
termination check is therefore **not a force cap**, and the simulated peak and
threshold must not be used as hardware-safety claims.

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
  --seed 1 --timeout-s 4 --allow-contact --open-loop-horizon 8 \
  --output results/my_pi05_trial
```

Omit `--allow-contact` to stop after detecting the first peg/fixture or
peg/table contact. This detection and the episode's provisional 5 N force
stop happen **after** a physics step: they cannot prevent contact or force
overshoot. The rollout starts with a grasped peg and executes the first eight
actions from each predicted 15-action chunk before taking another observation
and querying the policy. This matches the pinned OpenPI DROID example's
default open-loop horizon of eight; the horizon is configurable from 1 to 15.
Each action is clamped to joint limits and the target offset is limited to
0.1 rad/s divided by 15 Hz. This is **not** an actual joint-speed limit. An
existing saturated PD torque controller executes the targets; actual speed
and measured torque are not separately monitored for termination. The grasp
check requires contact with at least one finger, not proof of a stable
two-finger grasp.

The simulation is frozen while synchronous inference runs. The 15 Hz cadence
is simulation time, not a real-time guarantee; the client bounds its response
wait, but this is not a preemptive hardware watchdog. The rollout also stops
for release requests, stale simulation time, episode failure or timeout, and
clears arm controls on exit, including exceptions. These are development
guards, not safety certification. Never use this loop to operate a robot.
Rollout results report p50/p95/maximum inference latency and compare each
inference call against the chunk budget (eight action periods by default),
separately reporting the 15 Hz per-action period. The earlier 58-call pilot
re-inferred after every action: its p50 was 80.7 ms and p95 was 83.5 ms, so it
missed the 66.7 ms per-action period on every call. That was an unnecessarily
strict policy-query cadence, not how the pinned DROID example is run. The
chunked seed-1 rollout executed 60 actions over four seconds of simulation
time using eight warmed policy calls (p50 84.0 ms, p95 86.4 ms, max 86.7 ms);
none exceeded the 533 ms eight-action budget. Each inference still exceeds
the 66.7 ms single-action period, and the simulator pauses synchronously at
chunk boundaries, so this does not demonstrate a true 15 Hz wall-clock
control rate. The trial timed out and did not insert successfully. The first
cold request exceeded the default five-second client timeout; run the
non-actuating local-server preview once to warm the model before a rollout.

Collect a small set of classical impedance trials separately:

```sh
MUJOCO_GL=egl python3 collect_stage_c.py --output results/my_classical_pilot
```

The default seeds are train 0, 1, 2; validation 10; and test 11. Splits are
disjoint by seed. The manifest contains each outcome; `status: in_progress`
indicates an interrupted or failed collection, and the command reports the
error. Successful completion writes training-only means and standard
deviations for joint position, joint velocity, gripper-closed fraction, and
local wrench, with a 1e-6 standard-deviation floor. Validation and test trials
are excluded from the statistics. This file is auxiliary dataset metadata;
the pi0.5 adapter still passes raw inputs so OpenPI can apply the checkpoint's
own normalization transforms.

The `stage-c-development-v2` classical collector saves approximately 15 Hz
post-step camera/state samples and a separate `*-steps.npz` with every physics
step (currently 500 Hz). Each sample's `step_index` indexes the same-time
step. Step records include post-step simulation time, start time of the
preceding control interval, joint positions/velocities, full MuJoCo
`mjSTATE_INTEGRATION`, actuator controls and forces, raw local wrench and
peg/fixture contact force. The control is the input held over the immediately
preceding step; actuator force and sensors are evaluated at the post-step
state. The sample also contains the two RGB views, gripper fraction, hand
pose, peg/hole geometry and a depth-based phase label. The wrist wrench
includes gravity and inertia, not just contact. Phase is a heuristic and
outcome is per trial; the final valid step may precede an invalid-state
termination. Timing is simulation time, not wall-clock sensor timing.

The manifest records the MuJoCo version and a hash of the scene definition
and Panda assets to catch incompatible replay. It is not a full archive of
all dependencies or controller code. Older v1 datasets lack full-rate records;
they cannot be upgraded by interpolation. These classical actuator controls
are **not** DROID normalized velocity actions and are not training targets.

### Scripted simulation teacher

`integrations/scripted_teacher.py` provides an oracle teacher that reads
privileged MuJoCo peg/fixture geometry and emits a repeated 15x8 DROID
normalized joint-velocity chunk plus gripper-position targets (0=open,
1=closed). The rollout consumes its first row and maps each normalized joint
velocity to a per-command delta with DROID's 0.2-radian scale, then applies the
0.1 rad/s per-command bound, joint limits and PD target controller as other
policies. The teacher follows the successful position baseline's damped-IK
updates at 15 Hz. It is a synthetic simulator expert, not a human or physical
robot demonstration, and not evidence of transfer to a physical robot.

The teacher completed 10/10 local simulator checks over seeds 0–9 with
independent fixture offsets (±0.05 mm) and tilts (±0.03°); all met the
episode's insertion, clearance, angle and 5 N force-stop criteria. A
three-seed regression test also reconstructs each recorded joint target from
its normalized action and the observation-time joint positions, checking
that the labels match the bounded command actually executed. These tests
validate only this simulator/action pipeline. A separate collection is
described below; the teacher uses state unavailable to pi0.5, so it remains
synthetic oracle data rather than evidence of transfer.

Collect a larger, trial-separated oracle set (20 train, 5 validation, 5 test
trials by default) with disjoint randomized seeds. The collected
`results/stage_c_oracle_v2` set contains 2,466 observation/action samples:
1,641 train, 410 validation and 415 test. It also contains 2,046 complete
15-step action windows: 1,361 train, 340 validation and 345 test. All 30
trials passed the simulator success, geometry and force checks. The new
training split is about 5.9 times the 277 observations in the earlier
classical training set, and the compressed dataset is about 47 MB.

```sh
MUJOCO_GL=egl python3 collect_scripted_teacher.py \
  --output results/stage_c_oracle_v2
```

The collector fails closed if any trial does not satisfy the task's success,
depth, lateral-clearance, tilt and 5 N force-stop criteria. Each compressed
trial file stores the synchronized camera/state observation, the teacher's
normalized action, the action reconstructed from the bounded joint target
actually executed, and the resulting joint target. `action_chunk` stacks 15
consecutive executed 15 Hz actions, aligned with its recorded start
observation. Each normalized joint-velocity action maps to a joint delta by
multiplying by 0.2 rad, matching the pinned DROID Franka action adapter; the
gripper target and observation both use DROID's normalized position convention
(0=open, 1=closed). The exporter verifies the joint mapping against recorded
joint targets and checks the gripper state/target ranges. The local
0.1 rad/s command limit means joint labels are at most about 0.0333 in
magnitude, much narrower than DROID's full normalized range. The final 14
actions in each trial are retained as single-step labels but cannot start a
full window. The manifest records randomization, split seeds, action-label
contract, per-trial outcome and simulator provenance. The source's 15-step
chunks match the inference checkpoint horizon; fine-tuning uses per-frame
actions and the pinned `pi05_droid_finetune` configuration forms 16-step
windows. Convert the source data to LeRobot format before fine-tuning:

```sh
.openpi/.venv/bin/python export_oracle_to_lerobot.py \
  results/stage_c_oracle_v2 --output results/stage_c_oracle_lerobot
```

The exporter validates the source labels, timing, trial outcomes and DROID
action semantics, then writes disjoint local train/validation/test datasets.
It preserves the original 15 Hz normalized velocity labels and simulation
timestamps (the LeRobot frame timestamps are regularized to 15 Hz). OpenPI's
fine-tuning recipe must reuse the pretrained `pi05_droid` normalization assets.
The second external-camera feature is a black placeholder; the pinned
`DroidInputs` transform consumes external camera 1 and the wrist camera. The
data does not make the oracle's privileged state available to the policy or
establish representative real-world demonstrations.

The local conversion produced 20/5/5 episodes and 1,641/410/415 frames for
train/validation/test. Each split was reopened with the pinned LeRobot 0.1.0
reader, loaded as a 16-step action sequence, and passed through OpenPI's
`PromptFromLeRobotTask`, DROID repacking/input transforms, and 32-dimension
state/action padding. This verifies data-loader compatibility, not training
convergence or policy performance.

### Fine-tuning preflight and Alpine smoke test

`run_oracle_finetune.py` uses the pinned `pi05_droid_finetune` recipe with
only the converted `train` split and the pretrained DROID normalization
assets. It runs one complete OpenPI data-loader batch, including prompt
tokenization, before optionally starting training. The following is a
preflight only; it does not load model weights or train a policy:

```sh
OPENPI_DATA_HOME="$PWD/.openpi/model-cache" JAX_PLATFORMS=cpu \
  .openpi/.venv/bin/python run_oracle_finetune.py \
  --dataset results/stage_c_oracle_lerobot
```

The actual loader was verified locally with a `(2, 16, 32)` action batch.
LeRobot repeats the last action to pad sequences at episode boundaries;
the pinned OpenPI DROID repacking discards the padding mask, so these padded
labels enter the loss. This is upstream behavior, not an independently
validated terminal-action model.

For CU Boulder Alpine, the documented `aa100` partition offers both 40 GB
and 80 GB A100s. OpenPI estimates **over 70 GB** for full fine-tuning; use
an available 80 GB GPU for this full fine-tuning recipe. Availability,
allocation privileges and queue time must be checked on the cluster. OpenPI's
lower-memory LoRA option is a different experiment and is not configured by
this script. See the [Alpine hardware and GRES documentation](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
and [Slurm resource documentation](https://curc.readthedocs.io/en/latest/running-jobs/job-resources.html).

Before submitting, transfer the current uncommitted code and the converted
dataset to cluster storage, install the pinned OpenPI environment there, and
review the applicable checkpoint terms. The local `.openpi` environment and results
are not automatically included when cloning the repository. Prepare the
checkpoint/model cache on the cluster and keep weights private. An initial
single-node smoke-test request could use:

```text
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
```

These are starting estimates, not measured requirements or a submitted job.
The `--terms-reviewed` flag records the operator's acknowledgement; it does not establish
which terms apply or accept them on anyone's behalf. The 100-step run is
only a pipeline smoke test: OpenPI's unchanged learning-rate schedule has a
1,000-step warmup, so it is not a meaningful convergence experiment.
Serving a fine-tuned checkpoint is implemented but cannot be verified against
an actual trained checkpoint until one exists. Select experiments on
validation seeds and reserve test seeds for final evaluation. The older
single-seed rollout does not reproduce the teacher collection's fixture tilt
and must not be presented as a matched evaluation. Synchronous simulation
does not establish real-time operation or hardware safety.

#### Preparing Alpine after login

The local changes are uncommitted, so cloning this repository alone will not
transfer them. The steps below copy the current project and only the
converted training split. On Alpine, create a private workspace first:

```sh
umask 077
mkdir -p "$HOME/special-topics-private/project/results/stage_c_oracle_lerobot"
chmod 700 "$HOME/special-topics-private"
module load slurm/alpine
```

From the local repository root, copy the project and training data (requires
`rsync` locally and remotely):

```sh
rsync -av --exclude='.venv/' --exclude='.openpi/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='results/' \
  mujoco_force_control/ amhe1346@login.rc.colorado.edu:special-topics-private/project/
rsync -av mujoco_force_control/results/stage_c_oracle_lerobot/conversion_manifest.json \
  amhe1346@login.rc.colorado.edu:special-topics-private/project/results/stage_c_oracle_lerobot/
rsync -av mujoco_force_control/results/stage_c_oracle_lerobot/train \
  amhe1346@login.rc.colorado.edu:special-topics-private/project/results/stage_c_oracle_lerobot/
```

On the tested Alpine CPU node, glibc 2.28 cannot install the pinned
`rerun-sdk` wheel, which requires glibc 2.31 or newer. A Python 3.11 Bookworm
Apptainer image, OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`,
and its locked dependencies were installed on private Alpine scratch. The
data preflight timed out before it reported success; it must be retried.
Do not copy the host virtual environment into the container.

To update the already transferred project, run this from the local repository
root. It copies only the changed launcher and job template, not weights or
test data:

```sh
rsync -av mujoco_force_control/run_oracle_finetune.py \
  mujoco_force_control/alpine_pi05_smoke.slurm \
  amhe1346@login.rc.colorado.edu:/scratch/alpine/amhe1346/pi05-private/project/
```

In an allocated Alpine CPU session, retry with progress messages and a thread
dump every 60 seconds to locate any hang:

```sh
export WORK="/scratch/alpine/$USER/pi05-private"
export IMAGE="$WORK/containers/python311-bookworm.sif"
cd "$WORK/project"
apptainer exec --bind "$WORK:$WORK" \
  --env "OPENPI_DATA_HOME=$WORK/model-cache,JAX_PLATFORMS=cpu" \
  "$IMAGE" "$WORK/project/.openpi/.venv/bin/python" -u run_oracle_finetune.py \
  --diagnostic-interval-s 60 \
  --dataset results/stage_c_oracle_lerobot
```

The preflight loads data, normalization and tokenizer assets but not the
pretrained model weights. Check that the output says `preflight_passed`.
The presence and hash of Alpine's normalization file matched the locally
verified copy, but this alone does not prove that the loader succeeded.

The container smoke-job template requests an 80 GB A100. It checks that
checkpoint metadata exists and JAX sees an allocated GPU before training;
metadata presence does not establish that the weight files are complete or
uncorrupted. Prepare and independently verify the full pretrained `params`
cache before submitting. Submit from the scratch project directory only after
the data preflight and operator review of applicable terms:

```sh
export PRIVATE_WORK_DIR="/scratch/alpine/$USER/pi05-private"
export CONTAINER_IMAGE="$PRIVATE_WORK_DIR/containers/python311-bookworm.sif"
export EXPERIMENT_NAME=oracle-smoke-01
export TERMS_REVIEWED=yes
cd "$PRIVATE_WORK_DIR/project"
sbatch --account=ucb-general alpine_pi05_smoke.slurm
```

The OpenPI downloader changes cache-internal permissions to be broadly
accessible, so **the parent private directory is essential**; do not put the
cache directly in a shared directory. CURC recommends scratch for compute
jobs and may purge scratch data; use an appropriately private scratch directory
if available and back up needed checkpoints. The Slurm template and setup
have not been run on Alpine; resource availability, installation and account
access remain unverified.

### Matched policy evaluation

`evaluate_pi05.py` reconstructs the oracle collection's seeded lateral
offsets and fixture tilts and evaluates the server policy on the manifest's
validation split by default. It records each trial's fixture pose, result,
force, inference timing, policy-server metadata and simulator provenance in a
new output directory. It warms the model once without actuating and reports
an explicit error if the server or a trial fails. The policy runs 12-second
episodes with contact allowed and eight-action open-loop chunks; the oracle
teacher replanned every action, so this is not a direct matched-controller
comparison. Policy sampling is stochastic and its RNG advances across
warmup and trials; results are not a statistical performance estimate.
Use `--repeats N` to repeat each validation fixture without resetting the
policy RNG. Reports include per-seed outcomes and peak forces, plus per-trial
mean, p95, and time above the episode's reference contact force. Forces are
the sum of simulated peg-fixture and peg-table contact magnitudes, not a
hardware sensor reading.

Start the original checkpoint as described above, then in another terminal:

```sh
MUJOCO_GL=egl .openpi/client-venv/bin/python evaluate_pi05.py \
  --split validation --output results/my_pi05_validation
```

In the local original-checkpoint run, validation seeds 100–104 produced
**0/5 insertions**: four dropped pegs and one stopped before a requested
release. Peak recorded peg/environment contact forces per trial ranged from
0 to 1.793 N. All 67 warmed inference calls exceeded the 66.7 ms single
action period, but none exceeded the 533 ms eight-action budget (p50 83.0 ms,
p95 85.8 ms). The simulation pauses during inference, so this is not a
real-time demonstration. The cold, non-actuating warmup took 19.8 s. The
complete report is `results/stage_c_pi05_validation_baseline/report.json`;
the held-out test split was not used.

For a future fine-tuned checkpoint produced by `run_oracle_finetune.py`,
start the server with the checkpoint **step directory** (containing `params`
and `assets/droid/norm_stats.json`):

```sh
OPENPI_DATA_HOME="$PWD/.openpi/model-cache" XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  .openpi/.venv/bin/python run_openpi_local_server.py \
  --checkpoint /path/to/checkpoints/peg_oracle_pi05_droid/EXPERIMENT/STEP
```

Then run the same evaluation command with a **different** output directory.
The fine-tuning recipe predicts 16 actions; the server validates that shape
and exposes the first 15 to the existing simulation adapter. The first eight
are used per query, as for the original checkpoint. The server reports its
configured checkpoint and OpenPI revision, but metadata alone does not verify
the checkpoint contents. The adapter has been tested with mock outputs; a
fine-tuned checkpoint has not yet been trained or tested. Compare validation
results only under matching settings; use `--split test` only for final
held-out evaluation.

### Local observation and wall-paced probes

`inspect_demonstrations.py` reads only training trials, produces example
exterior and wrist images and reports action and image-change statistics:

```sh
.venv/bin/python inspect_demonstrations.py --output results/my_demo_inspection
```

The inspected 20 training trials contain 1,641 frames, all with closed-gripper
labels; the maximum absolute joint-action component was 0.01122. Example
camera views show the peg and fixture, but do not prove that the 80-micron
clearance or oracle action is inferable from images and joint state. The
teacher also uses privileged geometry and internal target history. Small
training variation and absence of policy recovery states limit what can be
concluded before training and validation.

For a separate wall-paced simulation probe, start the original local policy
server, then run:

```sh
MUJOCO_GL=egl .openpi/client-venv/bin/python probe_realtime_pi05.py \
  --seed 100 --output results/my_wall_paced_probe.json
```

This validation-only probe holds the previous joint target while inference
runs asynchronously and applies the returned chunk on control ticks. It
reports observation age, wall-clock action intervals and physics scheduling
lateness. It does not pipeline the next request during a chunk; rendering can
pause the simulator. A single local run dropped the peg after 8.906 simulated
seconds; 13 of 105 action intervals exceeded one 15 Hz period plus the 2 ms
physics-step tolerance, with a maximum interval of 200 ms. This is not a
15 Hz guarantee or a hardware test. Stochastic policy behavior and timing
may vary between runs.

Recorded pi0.5 trials additionally save `observations.npz` with the exact
preprocessed policy-input images and state at each inference time. The
manifest records the first clipped model action, the executed actions from
the configured open-loop horizon, their joint targets, observation time,
inference duration and outcome. Remaining actions in a model chunk are not
executed. The 10 Hz post-step samples and inference observations have
different cadences; join by simulation timestamp, not array index.
Exceptions produce an error manifest and are re-raised; a process
interruption can leave an in-progress manifest or partial files. New v2
pi0.5 recordings also include `steps.npz` at physics rate; earlier v1 files
do not.

### Evaluating imitation-label conversion

The tentative conversion derives a normalized joint action from *achieved*
motion: `(q_next - q_now) / (elapsed_seconds * 15 * 0.2)`. It uses future
state as an offline label, never as an online observation. It is not the
controller's commanded velocity. The validation tool restores each recorded
integration state independently and compares (1) replaying the recorded
per-step controls and (2) applying the converted command through the same
bounded joint-target/PD adapter as the pi0.5 rollout:

```sh
python3 validate_stage_c_conversion.py results/my_classical_pilot \
  --split train --output results/my_classical_pilot/conversion.json
```

`--refine-iterations 3` additionally tries fitting each bounded target using
the future endpoint and repeated local simulation. It is a diagnostic
experiment, not a trained policy, and cannot be used online. The report
includes joint and peg-tip endpoint errors, clipped intervals, and peak
replay force. This independent one-interval replay is not a continuous
closed-loop success or safety test; it does not apply episode termination
rules. Validate on held-out trials and full episodes before accepting labels.

On the three training trials in `results/stage_c_timing_provenance`, replay of
recorded controls reproduced both joint positions and peg tips exactly at
sample times. Direct converted labels had per-trial 95th-percentile tip errors
of 0.125–0.153 mm and maximum errors of 0.441–0.640 mm. Three refinement
iterations on the equivalent `results/stage_c_timing_pilot` trials reduced
the 95th percentiles to 0.043–0.100 mm, but maxima remained 0.258–0.364 mm
and four intervals hit bounds. This earlier dataset predates the provenance
check and cannot be passed to the current validation CLI. These errors are significant
relative to the nominal 0.08 mm radial clearance. Consequently neither
conversion is validated for imitation and **no imitation baseline has been
trained**; accurate action labels and continuous held-out replay are still
required.

One local development collection (`results/stage_c_development_v3`) produced
three training successes, one validation timeout and one test success; it has
277 training samples and is far too small for a meaningful performance claim.
In a separately recorded pi0.5 contact trial (`results/stage_c_pi05_recorded`),
seed 1 ended in a dropped peg at 3.806 s simulation time, with 2.208 N peak
peg contact and 58 inference calls of about 79–93 ms each. It was not a
successful insertion. A separate seed 0 no-contact trial stopped on contact
at 0.607 N after 1.282 s; it was recorded before inference-observation
logging was added. These examples are developmental, not performance or
real-time validation. Stage C is complete only as a **simulation integration
prototype**: preprocessing, bounded action postprocessing, local checkpoint
inference, chunked rollout, synchronized recording, training-only feature
statistics, and a multi-seed privileged scripted teacher are implemented and
tested. A 30-trial oracle dataset now provides 1,361 complete training action windows
(versus 277 earlier classical observation samples), with disjoint validation
and test splits. The converted data remains narrow and simulator-only;
successful policy training,
broader randomization and held-out policy evaluation remain open.
No successful pi0.5 insertion is recorded, and motion-derived classical
labels fail the clearance-scale replay check. The DROID-style eight-action
chunk cadence improves the inference budget, but synchronous simulation
pauses and this is not a real-time guarantee. The scripted teacher's
privileged state does not establish transfer. No physical robot was used, so
hardware validation is out of scope.

### pi0.5 checkpoint terms

The pinned OpenPI `pi05_droid` and `pi05_droid_finetune` configurations use
the default `gemma_2b` and `gemma_300m` variants, not the LoRA variants. The
model includes a PaliGemma component, and the same OpenPI revision contains a
2024 copy of `LICENSE_GEMMA.txt`, distinct from the OpenPI source-code Apache
license. Physical Intelligence explicitly lists `pi05_droid` for fine-tuning
and provides a custom DROID fine-tuning recipe.

Reviewed on 2026-11-17: the linked current
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) are dated April 1,
2026, and their appendix lists PaliGemma. They describe permitted use and
modification subject to the incorporated
[Prohibited Use Policy](https://gemma.ai/resources/prohibited-use/). If the
terms apply, redistribution or hosted access may require further downstream
restrictions, notices, and copies of the agreement; consult the applicable
version and its full provisions. Keep checkpoints private until those
obligations are resolved.

The published OpenPI recipe supports using this checkpoint for fine-tuning.
The materials reviewed did not establish the complete license for the
specific downloaded `pi05_droid` weight bundle or whether additional provider
terms apply; this is not evidence that private research fine-tuning is
prohibited. The model architecture and included Gemma terms alone do not
prove which agreement governs that bundle. Review the applicable terms before
training and seek clarification if needed, especially before sharing weights.
This review is not legal advice and does not accept terms on the operator's
behalf; the Apache code license alone does not settle weight rights.

An OpenPI contributor [stated in February 2025 that Apache 2.0 applies to
"everything"](https://github.com/Physical-Intelligence/openpi/issues/263#issuecomment-2638064959)
in response to a question about model weights. That statement predates the
pi0.5 release, and the repository later added a Gemma license, so it does not
by itself resolve which terms apply to this particular checkpoint.

[CURC's user policies](https://research-computing-user-tutorials.readthedocs.io/en/stable/additional-resources/policies.html)
require cluster-valid licenses for licensed software installed by CURC and
prohibit computational work on login nodes. They do not confirm this model's
license or this project's Alpine eligibility. Ask Physical Intelligence about
the checkpoint and applicable component terms if clarification is needed;
consult [CURC](https://www.colorado.edu/rc/contact) or an institutional
reviewer about private checkpoint storage, access and use on Alpine as needed.

Verification references, pinned to OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`:

- [Official DROID checkpoint and serving instructions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README.md)
- [DROID rollout and eight-action open-loop default](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)
- [`pi05_droid` model configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)
- [DROID input/output transformation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)
- [Gemma terms included by OpenPI](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt) and [current Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- [OpenPI source license](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE)
- [DROID gripper conventions](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py) and [joint-action scaling](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py)
- [OpenPI requirements and checkpoint list](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md), [code license](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE), and [custom DROID fine-tuning guide](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README_train.md)

OpenPI's code is Apache-2.0 licensed, while the `pi05_droid` model
configuration uses Gemma variants and the upstream repository ships Gemma
terms. This identifies the relevant published terms but does not confirm
acceptance on a particular checkpoint download or rule out additional
provider terms; see **pi0.5 checkpoint terms** above. OpenPI lists an NVIDIA
GPU with over 8 GB memory for inference; availability and actual memory use
must be checked on the machine running the server.

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