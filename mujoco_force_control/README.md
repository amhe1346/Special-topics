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