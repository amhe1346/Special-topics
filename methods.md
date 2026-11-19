# Methods: Force-Aware Residual Control for Peg Insertion

**Working overview, 19 November 2026.** This document separates methods already
used in simulation from proposed methods. It is an outline for a longer
phase-by-phase write-up, not a report of completed residual-policy results.
Descriptions and video plans should be updated as experiments are performed.

## Research question and study scope

The project asks whether force information in a residual policy's observations
and force-aware reward shaping independently improve peg-in-hole insertion
success, and whether their effects interact, when the same pi0.5
vision-language-action policy supplies the nominal action.

The current study is **simulation-only**. The broader question of performance
on a physical robot remains a future research question: the present work
does not validate physical sensors, control, or safety. The planned comparison
uses a fixed nominal pi0.5 policy and varies force observation and reward
independently. The current fine-tuning smoke test is preparation for a nominal
policy; it is not itself a test of the force-aware residual hypothesis.

## Organization

The research stages describe the experimental sequence. The implementation
phases describe tools built along the way and do not map one-to-one onto those
stages.

| Research stage | Methods and relationship to implementation phases | Status |
|---|---|---|
| A. Task and protocol | Define the simulated insertion task, outcomes, stopping rules, and data contract; uses the Phase 0 simulation foundation. | Defined for simulation. |
| B. Benchmark and classical baselines | Build and test the task, sensing and controllers; draws on Phases 1–5 and 7. | Simulation benchmark prototype implemented. |
| C. pi0.5 and demonstrations | Connect images and robot state to pi0.5, collect scripted demonstrations, and prepare a nominal imitation policy. | Integration and dataset prepared; checkpoint saving and trained-policy evaluation remain unconfirmed. |
| D. Residual policy | Add bounded corrections and compare force-observation and reward factors. | Planned; not yet implemented. |
| E. Evaluation | Compare variants with nominal-only and classical baselines on defined simulation conditions. | Baseline evaluation exists; residual-policy comparison is planned. |
| F. Analysis and reporting | Analyze success, contact, failures and the two experimental factors. | Protocol planned; final comparative analysis not yet performed. |

Phase 6 provides simulated CAN-bus and virtual-MCU infrastructure. It is not
currently evidence of physical communication or a demonstrated part of the
pi0.5 residual-policy experiment. Describe it separately if included.

## Stage A: Simulated task and protocol

The task places a grasped peg above a modeled hole in a MuJoCo Franka Panda
scene. The nominal peg diameter is 8 mm and hole diameter is 8.16 mm, giving
0.08 mm nominal radial clearance for aligned rigid geometry. The modeled hole
uses faceted walls. Trials begin with the peg already grasped; pickup is not
part of the protocol.

Success is determined independently of a learned reward using simulator
geometry: entrance crossing, insertion depth, lateral and angular alignment,
clearance along the peg, and a continuous seated dwell. Contact force, drops,
workspace violations and other termination reasons are also recorded. The
simulation's contact-force stop is checked after a physics step and can
overshoot; it is not a physical safety guarantee.

**Connection to the question:** a common task and independent success measure
make it possible to compare residual variants without defining success by the
same force-aware reward being tested.

## Stage B: Sensing, control and benchmark

The simulator provides joint-state sensing, joint-space position control,
actuator-force feedback, torque observers, and a simulated six-axis hand
load-cell signal. Classical force and compliant-control methods were
implemented as simulation tools. A peg-in-hole benchmark applies position,
hybrid position/force, and impedance controllers to seeded task conditions and
records success, termination reasons, and force summaries. The benchmark can
add seeded noise and fixed latency to the simulated wrench.

The hand load cell measures a transmitted local wrench. It is distinct from
the evaluator's ground-truth peg/environment contact-force metric. Both are
simulated, and their relationship to a physical sensor is not established.

**Connection to the question:** the benchmark provides common trial conditions
and classical reference methods; simulated wrench sensing provides a candidate
input for later force-observation experiments.

## Stage C: Nominal pi0.5 policy and demonstrations

The pi0.5 integration converts simulated camera images and robot state into
the policy's input and applies bounded postprocessing to its action chunks.
A scripted teacher generated demonstrations using privileged simulator
geometry and internal target history. The converted training split contains
20 successful episodes and 1,641 frames, with separate validation and test
fixtures. Training uses the pretrained DROID normalization assets. These are
synthetic, narrowly distributed demonstrations; successful teacher execution
does not establish that the camera-based policy can reproduce it.

An original-checkpoint validation run produced 0 successes in five trials.
One 100-step GPU training attempt reached checkpoint saving at step 50 but
was killed for exceeding system memory. A higher-memory retry was running
at the last check; a completed, loadable trained checkpoint is not yet
confirmed. The smoke test is for pipeline validation, not convergence.

**Connection to the question:** a verified nominal policy is needed before
the effect of a residual correction can be assessed. Training pi0.5 itself
does not answer whether force feedback helps.

## Stage D: Planned residual-policy experiment

After selecting a nominal checkpoint, the proposed residual policy will
produce a bounded correction to pi0.5's action. The initial interface design
below has been selected. The wrench preprocessing helper and residual adapter
are connected to an opt-in simulation rollout and tested with mock policies.
They are not connected to a trained residual policy or the existing evaluation
command. Pilot reward accounting and an exploratory mock-only PPO smoke test
are connected to that rollout; a real pi0.5 residual-training experiment is
not yet implemented.

### Initial residual interface

- **Nominal policy:** pi0.5 receives its existing camera and robot-state
  observations and produces a chunk of nominal actions. The current rollout
  executes up to eight actions before requesting a new chunk.
- **Update rate:** the residual updates at each 15 Hz simulation control tick,
  using the current action from the nominal chunk. It may therefore respond to
  new robot state or wrench samples between pi0.5 queries. This is a
  simulation-time design, not a demonstrated wall-clock control rate.
- **Residual observation:** current joint positions and velocities, gripper
  state, current nominal action, and a simulation timestamp. The residual
  does not receive camera images in this initial design. Force-on variants
  additionally receive the latest available six-axis wrist wrench and its
  sample timestamp; force-off variants receive no wrench or derived force
  features. Robot state can still change in response to contact in all
  variants. No privileged peg or hole geometry is supplied to the residual.
  The PPO smoke encoder omits both timestamps and the nominal gripper flag
  (always closed whenever the residual policy is called).
- **Force signal:** the simulated load cell's sensor-local force in newtons
  and torque in newton-metres, ordered force XYZ then torque XYZ. It is not
  the evaluator's separate peg/environment contact-force metric. For initial
  experiments, follow the benchmark's per-trial initial wrench tare, then add
  independently sampled Gaussian noise with standard deviations 0.05 N per
  force axis and 0.005 N m per torque axis, and delay the sample by two
  physics steps. Use seeded noise, retain the original sample timestamp, and
  apply no additional filter. The tare is a simulated baseline subtraction,
  not physical sensor calibration. Fit any normalization on training data
  only and save its statistics.
- **Residual output:** seven joint-position corrections in radians per control
  tick, in the same joint order as pi0.5. The residual does not modify the
  gripper command; pi0.5's existing gripper and release handling remains in
  effect.
- **Combination:** start with the nominal joint delta after the existing
  per-tick speed clamp; add the bounded residual delta; then reapply the same
  total per-tick speed clamp and joint-range clamp before executing the
  target. With a zero residual, the resulting target must match the existing
  nominal-only rollout. Initially bound each correction to +/-0.001 rad per
  tick, approximately 15% of the existing 0.1 rad/s speed clamp divided by
  15 Hz. This is a provisional development setting, not a physical limit or
  an optimized value; use the same bound across variants.
- **Validation and records:** reject non-finite or malformed inputs and
  outputs and do not silently substitute a zero correction on errors. Force-on
  trials must stop with an explicit sensor-related reason if a wrench sample
  is unavailable, non-finite, future-dated, or more than 0.1 simulation
  seconds old; do not silently substitute zero or hold an expired sample.
  Allow the delay buffer to fill before residual control begins, and stop if
  it cannot supply a valid sample at the first control tick. Apply the same
  sensor-health gate across variants for comparable conditions, but provide
  neither wrench values nor derived force features to force-off residuals.
  Record this common termination gate as a potential indirect effect of
  sensor health. Log timestamps, nominal, residual, and final executed
  actions, clipping, and sensor status for each tick. Apply the same
  non-force interface and execution limits to all four variants.

The existing simulator's post-step force stop and actuator limits are not
physical safety guarantees, and this adapter does not by itself provide an
independent torque or actual-speed safety system. Mock tests cover zero-residual
equivalence, bounds, update timing, force isolation, and error handling,
including opt-in mock rollouts. Testing with a real nominal checkpoint and a
trained residual policy remains before any performance claim.

A separate classical position-controlled insertion was used as a contact
probe with the residual adapter running in shadow; the adapter did not drive
the robot. In paired seed-0 trials, the measured sensor delay was two 0.002 s
physics steps. The force-enabled shadow policy received wrench observations
on all 75 calls, while the force-off shadow policy received none. Peak
peg/environment contact force was 0.388 N. The maximum tared wrist-force
magnitude was 0.183 N for samples with contact force above 0.1 N, versus
0.069 N for samples at or below that diagnostic threshold. This single
simulated trial checks the wiring, not a calibrated sensor or a causal
force-to-wrench relationship.

The factorial variants are defined as design-only configurations:

| Variant | Wrist wrench in residual observation | Force-aware reward |
|---|---|---|
| Baseline residual | Off | Off |
| Observation only | On | Off |
| Reward only | Off | On |
| Combined | On | On |

The configurations select the adapter's force-observation flag and the
pilot reward factor in opt-in simulation rollouts. No training runner exists;
the four variants have not been trained.

### Selected reward and training direction

The selected pilot direction is a shared task-progress reward plus a terminal
success reward. Reward-on variants will additionally penalize simulated
peg/environment contact force above the existing 1 N analysis reference.
The force term must not penalize all contact: insertion requires contact.
The 1 N reference is an analysis threshold, not the 5 N post-step termination
threshold and not a physical safety limit. Reward calculations may use
simulator ground truth, but force-off residual observations must not receive
the contact force, its derived features, or reward values as input.

The provisional potential is `0.5 * exp(-lateral_error / 0.00008 m) +
0.5 * clip(depth / 0.020 m, 0, 1)`, and its change has weight 1. Add 10
for evaluator success and subtract 5 for scored failures. For reward-on
variants subtract the physics-step integral of force above 1 N, with weight
1 reward unit per N s. The selected PPO pilot discount is 0.99 per decision
step. A partially executed terminal action counts as one terminal decision
step with no terminal bootstrap; preceding full decision transitions use
gamma 0.99. Invalid simulator, sensor, or inference states should abort and
report the training run without scoring the invalid transition. These
training semantics are implemented in the mock-only PPO smoke test, not yet
in a real training pipeline. Invalid simulation or sensor outcomes are not
scored by the calculator. This
undiscounted potential difference is not guaranteed to preserve the optimal
policy under discounting. The residual rollout now records reward breakdowns
over executed physics samples, with no discount calculation. A nominal release
is a separate zero-duration terminal event, not a residual action. For PPO,
its -5 is added to the preceding residual-action reward, with no extra
discount step. If release occurs before any residual action, the -5 remains
an explicitly reported zero-action failure and supplies no PPO sample. If
the entire batch has no residual actions, the smoke test aborts without an
update. An invalid trial has no scored total and any incomplete invalid
interval is not scored.

Offline scoring of a seed-0 classical success and a deliberately shortened
timeout gave undiscounted totals of approximately 10.346 and -5.105; neither
activated the force penalty. The scoring begins after the first physics step
and uses intervals anchored to that first recorded sample. A separate
kinematic jam probe exceeded the force reference, but deliberately continued
past normal stopping and is not representative of a training episode. These
are diagnostic checks, not evidence that the weights are effective.

An additional classical position trial at seed 2 and a wider +/-5 mm fixture
offset range stopped normally at the force limit. One 0.002 s sample exceeded
1 N, peaking at 51.964 N despite the post-step 5 N stop. The integrated
excess was 0.101928 N s, resulting in a -0.101928 force term; total scores
were -4.325 without and -4.427 with that term. The trial's initial conditions
differ from the seed-0 success and timeout, so their totals are not paired
comparisons. The small integrated penalty relative to the -5 failure penalty
shows that this pilot term may weakly penalize brief force spikes; more
terminated trajectories and an explicit peak-force objective may be needed
before choosing weights.
The integral-only force term remains selected for initial diagnostics; no
separate peak-force penalty has been adopted.

The selected learner to investigate is PPO with the selected pi0.5 nominal
checkpoint frozen during residual training. A small PPO smoke test now collects
mock-nominal episodes and performs one update; it is not a full training
runner or an evaluation of learning. Before adopting a budget, measure simulation and nominal
inference throughput, memory use, and available compute; then specify a numeric
budget and multiple training seeds. Match environment steps, optimization
schedule, seeds, scenarios and resource reporting across the four variants.
Force-on and force-off networks have different input dimensions; document
architecture and parameter counts rather than assuming they are identical.
The existing synchronous inference pauses simulation, so throughput estimates
do not establish real-time operation. Evaluate checkpoints on validation
conditions and keep test conditions reserved.

As an initial plumbing measurement, three one-second simulated episodes per
variant with mock nominal and residual policies achieved approximately
33.1–34.0 action steps per wall-clock second locally, including setup and
settling. All episodes timed out; peak process RSS for the complete run was
about 759 MiB. This small, sequential mock benchmark excludes pi0.5 inference,
PPO optimization and GPU memory, and cannot determine the PPO budget. Measure
with the actual nominal checkpoint and training implementation before choosing
that budget.

Separately, a new local probe of the original cached pi0.5 DROID checkpoint
used one non-actuating warmup followed by a seed-1, four-second synchronous
simulation rollout. It timed out after 60 actions; eight warmed inference
calls had p50 84.5 ms, p95 86.4 ms, and maximum 86.9 ms. The local RTX 4090
reported 20,227 MiB total GPU memory in use after warmup, with the server
configured to preallocate 80% of GPU memory. That is a device-wide snapshot,
not peak model memory, and includes other GPU usage and allocator reservation.
The server was shut down after the probe. This is the original checkpoint,
not a verified fine-tuned Alpine checkpoint; server metadata does not
independently verify checkpoint contents. The single rollout and warmup do
not measure PPO updates or establish a matched training budget.

For the smoke test, fixed observation scales are joint positions divided by
pi radians, velocities by 1 rad/s, gripper fraction unchanged, and nominal
joint deltas divided by 0.1/15 radians. Force-on inputs append wrist force
divided by 5 N and torque by 1 N m; force-off inputs have no wrench fields.
These are provisional numerical scales, not clipping, calibration or safety
limits. The two 32-unit tanh actor/critic networks use a Gaussian latent
action mapped through tanh to the +/-0.001 rad bound; PPO scores the sampled
latent action, not the subsequently clipped joint target. The smoke update
uses gamma 0.99, GAE lambda 0.95, clip 0.2, two full-batch epochs and Adam
learning rate 0.0003. Two 0.3-second mock episodes per variant all timed out
and yielded ten action samples each; this checks the update path, not whether
the policy learns insertion. No checkpoint is saved.

The same smoke collector was also tested with the original cached pi0.5
checkpoint as a separately served, frozen nominal policy. The baseline and
combined variants each collected two 0.3-second episodes and ten actions;
all timed out and both residual actors changed during a single update.
No force-excess penalty was activated in those episodes. Separate tests
inject force samples to verify reward-on and reward-off accounting through
the collector, and exercise release and invalid-sensor termination. This is
an integration check, not a learned-policy comparison. The Alpine fine-tuned
checkpoint remains unverified, and server metadata is not independent proof
of checkpoint contents.

A future resumable trainer should checkpoint at completed update and episode
boundaries, recording actor, critic, distribution, optimizer and scheduler
state, counters, configuration, reward and observation scales, seeds and all
relevant RNG states, and verified nominal-checkpoint and simulation/code
provenance. Publication should be atomic; incompatible or incomplete files
must be rejected, and an uninterrupted versus restored run must be tested.
The nominal server may sample independently: exact resume cannot be claimed
without saving and restoring its RNG state too. This is a proposed contract,
not implemented checkpointing or a chosen checkpoint cadence.

The nominal checkpoint, non-force inputs, action interface, control checks,
training budget and evaluation conditions should be held consistent across
variants. Force-aware reward should not simply discourage all contact:
insertion requires contact. Stopping rules remain independent of reward.

**Connection to the question:** the 2-by-2 design separates the effect of
observing force from the effect of force-aware reward, and permits their
interaction to be investigated.

## Stages E and F: Planned evaluation and analysis

Once trained policies exist, evaluate the four variants along with the
nominal-only policy and classical baselines on matched simulation scenarios.
Use validation conditions for development and model selection and reserve
the test split for final evaluation. Record training seeds and trial
conditions; use paired scenarios where practical.

Report success and the individual geometric checks, failures and termination
reasons, peak and time-varying simulated force, and timing. Analyze the
observation and reward factors and their interaction with uncertainty
appropriate to the number of training seeds and scenarios. Small pilot runs
support feasibility and debugging, not broad generalization claims.

**Connection to the question:** matched comparisons and independent outcome
measurements make it possible to distinguish a useful force-informed
correction from improvements caused by other changes to the experiment.

## Video storyboard

No videos have been recorded for this write-up yet. Each clip should show an
actual run or existing recorded data, identify the controller and seed, and
state whether it is simulation footage, a visualization of recorded data, or
a planned illustration. Show failures and force overshoots when relevant;
do not use a successful clip as evidence of overall success rate.

| Section | Proposed clip | Evidence to accompany it | Availability |
|---|---|---|---|
| A | Show the peg, hole, cameras and initial grasp, then explain the geometric success criteria. | Scene dimensions and evaluator rules. | Scene exists; recording still needed. |
| B | Show one seeded classical-controller insertion, paired with its actual force trace and outcome. | Controller, seed, settings, termination and trace provenance. | Benchmark and interactive viewer exist; synchronized presentation and recording still needed. |
| C | Show the teacher's camera observations and a scripted insertion; separately illustrate a recorded pi0.5 validation outcome. | Dataset manifest, model identity and actual trial outcome. | Saved camera examples and evaluation reports exist; videos still needed. |
| D | Show nominal and residual actions and wrench response during a real residual-policy run. | Variant configuration, action bounds and synchronized wrench/action log. | Future work: policy not implemented. |
| E–F | Show actual matched outcomes and measured force comparisons after experiments. | Complete trial reports, seeds and analysis. | Future work: comparative results not available. |

For each completed clip, retain the command or script, software revision,
seed, settings, data source, and outcome alongside the video. A short clip
illustrates a method or individual trial; it does not replace the full
evaluation or its failed runs.

## Source material

- [Simulation methods, benchmark and pi0.5 evaluation](mujoco_force_control/README.md)
- [Research outline and staged protocol](thoughts)
- [Demonstration inspection](mujoco_force_control/results/stage_c_demonstration_inspection/report.json)
- [Example training camera views](mujoco_force_control/results/stage_c_demonstration_inspection/training_camera_examples.png)
- [Original pi0.5 validation report](mujoco_force_control/results/stage_c_pi05_validation_baseline/report.json)
