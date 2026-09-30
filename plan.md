# Plan: MuJoCo Closed-Loop Position/Force Control Learning Project

## Goal
Staged learning project in MuJoCo (Python bindings) building up to a hybrid position/force
controller for a multi-DOF robotic arm, using encoder (joint pos/vel), load-cell (F/T sensor),
and motor current (actuator force) sensing — all via MuJoCo's built-in sensors as stand-ins.

## Decisions
- Arm model: MuJoCo Menagerie Franka Emika Panda (has torque-controlled actuators; swap-friendly
  with UR5e later since code is written against generic joint/actuator names).
- Sensors: use MuJoCo native `jointpos`/`jointvel` (encoder), `force`/`torque` site sensor at
  end-effector (load cell), `actuatorfrc` (motor current proxy for torque).
- Control progression: PD position (encoder) → current-based torque estimate refined by EKF/SMO
  friction observer → + end-effector load-cell force feedback → three standalone comparable
  paradigms (Hybrid Position/Force, Cascaded Force w/ inner torque+position loops,
  Impedance/Admittance) → benchmark comparison across paradigms → distributed MCU comms.
- Torque estimation: naive `tau = Kt * I` (via `actuatorfrc`) is inaccurate due to cogging torque
  and harmonic-drive friction — fuse motor current with high-frequency encoder acceleration using
  both an Extended Kalman Filter (EKF) and a Sliding Mode Observer (SMO) to subtract internal
  friction and isolate true environmental contact force ahead of (faster than) load-cell readings.
- Paradigm comparison: implement all three as standalone controllers, benchmark under identical
  test scenarios/disturbances, log standardized metrics (tracking error, force error, settling
  time, overshoot, energy/effort).
- Language: Python, official `mujoco` package + `mujoco.viewer` for visualization.
- Project structure: proper Python package with venv, `requirements.txt`, `envs/`,
  `controllers/`, `notebooks/`, `tests/`.

## Steps

### Phase 0 — Project scaffolding
1. Create project folder structure:
   - `mujoco_force_control/` (root)
     - `envs/` — MJCF/XML scenes + Python env wrappers
     - `controllers/` — controller implementations per stage
     - `sensors/` — sensor-reading + filtering/noise utilities
     - `notebooks/` — exploratory Jupyter notebooks per stage
     - `tests/` — pytest sanity checks (simulation stability, controller convergence)
     - `requirements.txt` (mujoco, numpy, matplotlib, jupyter, pytest)
     - `README.md` with stage overview
2. Set up Python venv, install `mujoco`, pull Franka Panda MJCF from MuJoCo Menagerie
   (vendor into `envs/assets/panda/` or add as git submodule). *depends on 1*
3. Write a minimal `envs/base_env.py` wrapper: load model/data, step sim, expose sensor reads
   (`data.sensordata` indexed by named sensors defined in MJCF `<sensor>` section). *depends on 2*
4. Verify: load model, run a few sim steps with zero control, render via `mujoco.viewer`,
   confirm gravity-compensated arm sags/holds as expected. *depends on 3*

### Phase 1 — Encoder-based position control
5. Add MJCF `<sensor>` entries: `jointpos`/`jointvel` per joint (this is the "encoder").
6. Implement PD (or PID) joint-space position controller in
   `controllers/pd_position_controller.py`, reading encoder sensordata instead of raw `qpos`.
7. Notebook/script: command step/sinusoid joint targets, plot tracking error, tune gains.
8. Verify: tracking error converges within tolerance for step and trajectory targets; pytest
   check on settling time/overshoot bounds.

### Phase 2 — Motor current (torque) sensing
9. Add `actuatorfrc` sensor per actuator (proxy for motor current via torque constant).
10. Extend controller to log/use actuator force feedback: implement simple disturbance/gravity
    compensation check — compare commanded vs. sensed torque. *depends on Phase 1*
11. Verify: sensed actuator force matches expected gravity/inertial torque in static poses
    (cross-check against `mujoco.mj_inverse` as ground truth).

### Phase 2.5 — Sensor Fusion & Observer Design (EKF + SMO)
11a. Implement `sensors/torque_observer_ekf.py`: Extended Kalman Filter fusing motor current
     (`actuatorfrc`) with high-frequency encoder acceleration (finite-difference of `jointvel`,
     or a second `jointactuatorfrc`/accel sensor) to estimate and subtract cogging-torque and
     harmonic-drive friction, isolating true environmental contact torque. *depends on Phase 2*
11b. Implement `sensors/torque_observer_smo.py`: Sliding Mode Observer for the same estimation
     task, as a more robust (but chattering-prone) alternative. *depends on Phase 2*
11c. Inject synthetic friction (nonlinear joint friction/cogging terms in MJCF or applied as
     extra disturbance torque) so the observers have something nontrivial to reject; compare
     EKF vs. SMO estimation error against ground-truth contact torque (known in sim).
11d. Verify: both observers reduce torque-estimate error vs. naive `tau = Kt * I` baseline;
     log convergence speed and steady-state bias for EKF vs. SMO; observer output becomes the
     torque/force signal consumed by Phase 3 onward (replaces raw `actuatorfrc` scaling).

### Phase 3 — Load-cell force sensing at end-effector
12. Add a `<site>` at end-effector + MJCF `force`/`torque` sensor pair (6-axis load cell stand-in).
13. Implement `controllers/force_controller.py`: task-space force control (map desired
    end-effector force → joint torques via Jacobian transpose), closed loop on load-cell reading,
    fused with the Phase 2.5 observer estimate to detect contact ahead of load-cell response.
14. Verify: arm pushes against a fixed obstacle/wall body in the scene, achieves commanded
    contact force within tolerance; plot force tracking; confirm observer-based estimate flags
    contact onset earlier than raw load-cell signal (the "faster than load-cell" claim).

### Phase 4 — Comparable control paradigms (A/B/C)
15. Implement paradigm A — `controllers/hpfc_controller.py` (Hybrid Position/Force Control):
    selection matrix splitting task space into position-controlled and force-controlled axes;
    combines encoder-based position loop (Phase 1) and load-cell/observer force loop (Phase 3).
    *depends on Phases 1 & 3*
16. Implement paradigm B — `controllers/cascaded_force_controller.py` (Force Control with Inner
    Torque & Position Loops): outer force-error loop produces a position/torque setpoint, inner
    loop is a fast torque control (using Phase 2.5 observer estimate) nested inside a position
    loop — classic cascaded architecture. *depends on Phases 1, 2.5 & 3*
17. Implement paradigm C — `controllers/impedance_controller.py` (Impedance/Admittance Control):
    regulates the dynamic relationship (virtual spring-damper) between position error and
    contact force rather than tracking an explicit force target; uses all three sensor
    modalities (encoder, observer-based torque, load cell). *depends on Phases 1, 2.5 & 3*
18. Verify: scenario where arm free-moves in unconstrained axes while maintaining constant
    contact force in a constrained axis (e.g., wiping a surface) — run for A, B, and C
    individually and confirm each is independently stable/convergent.

### Phase 5 — Wrap-up
19. Write `tests/` pytest suite covering Phases 1–4 controllers (import model, run N steps,
    assert convergence bounds) so regressions are caught.
20. Write `README.md` summarizing each stage, how to run notebooks/scripts, and key MuJoCo
    concepts learned (sensors, actuators, Jacobians, selection matrices, observers).

### Phase 6 — Distributed MCU communication (simulated CAN bus)
20. Design `comm/sim_bus.py`: shared message-queue bus model with per-message transmission delay
    (bitrate-based), configurable latency/jitter, and CAN-style priority arbitration (lower ID
    wins on contention). Pure Python, no hardware dependency.
21. Define `comm/mcu_node.py`: one virtual MCU per joint/actuator — owns that joint's encoder
    (`jointpos`/`jointvel`) and motor-current (`actuatorfrc`) sensor reads (from Phases 1 & 2),
    publishes periodic state frames onto the bus, subscribes to torque-command frames addressed
    to it. *depends on Phase 2*
22. Add a central "controller node" (`comm/controller_node.py`) that runs one of the Phase 4
    paradigm controllers (A/B/C, selectable) but only sees sensor data arriving via the bus (with
    latency) rather than direct `data.sensordata` access — forces controller to be written
    against stale/delayed state. *depends on Phase 4, step 21*
23. Wire load-cell (end-effector F/T) readings through a dedicated MCU node as well, so the
    controller under test consumes all three modalities (encoder, current/observer estimate,
    load cell) purely via bus messages. *depends on 21, 22*
24. Verify: inject bus latency/bitrate limits and confirm controller performance degrades
    gracefully (increase PD gains cause instability at high latency — demonstrates why comms
    design matters for closed-loop control); log per-node message timestamps to check arbitration
    priority behaves as expected under simulated bus contention (multiple nodes transmitting same
    step).
25. Optional stretch: swap `sim_bus.py` backend for real serial/CAN hardware (e.g., python-can)
    behind the same interface, so code could later target actual MCUs without controller changes.

### Phase 7 — Paradigm comparison benchmark
26. Build `benchmark/harness.py`: runs each of the Phase 4 paradigms (A: HPFC, B: Cascaded
    Force/Torque/Position, C: Impedance/Admittance) against identical test scenarios (same
    target trajectory, same injected disturbance/contact event, same sim seed).
    *depends on Phase 4 (steps 15-18)*
27. Define standardized metrics logged per run: position tracking error (RMSE), force tracking
    error (RMSE), settling time, overshoot %, control effort/energy (sum of |torque| or
    torque²·dt). Store results in `benchmark/results/` as CSV.
28. Generate comparison plots/table (`benchmark/compare.py`): overlay tracking/force curves for
    A vs B vs C, bar chart of metrics side by side.
29. Verify: benchmark runs reproducibly (same seed → same metrics); results clearly show
    tradeoffs (e.g., C smoother but less precise force tracking than A, B faster disturbance
    rejection due to inner torque loop with observer feedback from Phase 2.5).

## Relevant files
- `mujoco_force_control/sensors/torque_observer_ekf.py` — EKF fusing current + encoder accel
- `mujoco_force_control/sensors/torque_observer_smo.py` — Sliding Mode Observer alternative
- `mujoco_force_control/controllers/hpfc_controller.py` — paradigm A (hybrid position/force)
- `mujoco_force_control/controllers/cascaded_force_controller.py` — paradigm B (inner torque+position loops)
- `mujoco_force_control/controllers/impedance_controller.py` — paradigm C (impedance/admittance)
- `mujoco_force_control/benchmark/harness.py` — runs A/B/C under identical scenarios
- `mujoco_force_control/benchmark/compare.py` — metrics table + comparison plots
- `mujoco_force_control/comm/sim_bus.py` — simulated CAN bus (latency, bitrate, arbitration)
- `mujoco_force_control/comm/mcu_node.py` — per-joint virtual MCU (encoder + current sensing, tx/rx)
- `mujoco_force_control/comm/controller_node.py` — central controller consuming bus-delivered state
- `mujoco_force_control/envs/assets/panda/panda.xml` — Menagerie MJCF, add `<sensor>` block here
- `mujoco_force_control/envs/base_env.py` — shared sim stepping + sensor access wrapper
- `mujoco_force_control/controllers/pd_position_controller.py`, `force_controller.py` — earlier-stage controllers
- `mujoco_force_control/tests/test_controllers.py` — convergence/stability assertions

## Verification
1. Each phase has its own runnable script/notebook producing tracking-error or force-error plots.
2. `pytest tests/` passes for all implemented controller stages.
3. Manual: `python -m envs.base_env --viewer` opens interactive MuJoCo viewer to visually confirm
   arm behavior at each stage.
4. Phase 6: unit tests for `sim_bus.py` arbitration/latency correctness; end-to-end run showing
   the tested controller still converges (within relaxed tolerance) when running over the
   simulated bus vs. direct sensor access baseline from Phase 4.
5. Phase 2.5: EKF/SMO observer estimation error vs. ground-truth contact torque is lower than
   naive `Kt * I` baseline; both observers detect contact onset before the load-cell sensor does.
6. Phase 7: benchmark harness produces reproducible metrics (same seed → same numbers) and a
   comparison table/plot clearly differentiating A/B/C tradeoffs.

## Further Considerations
1. Menagerie Panda vendoring: git submodule vs. vendored copy — recommend vendoring (copy files
   in) to avoid submodule friction for a learning repo; can revisit if arm model needs updates.
2. Real sensor realism (noise/quantization/delay) was explicitly deferred (using clean MuJoCo
   sensors) — can be added later as a "sim-to-real gap" stretch phase if desired.
