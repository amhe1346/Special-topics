Summary of today’s work
Today’s main focus was preparing Stage D’s residual-policy training pipeline while the Alpine fine-tuning job remains queued.

What we worked on
Connected the simulated wrist-wrench input, residual adapter and provisional reward to a mock simulation rollout.
Checked force signals in a controlled insertion and kept the wrist wrench separate from the evaluator’s peg/environment contact-force metric.
Defined the four experiment variants: force observation on/off × force-aware reward on/off.
Added reward accounting for action intervals, terminal outcomes and force above the 1 N analysis threshold.
Measured local mock-rollout throughput: approximately 33–34 action steps per wall-clock second across variants. This does not include real pi0.5 inference or PPO training.
Built a small PPO smoke test and connected it first to mock nominal actions, then to the original pi0.5 checkpoint. Both force-off and force-on integration runs completed an update, but all short episodes timed out.
Documented a proposed checkpoint-and-resume contract for a future trainer. Saving and resuming are not implemented yet.


Problems and fixes
Problem	What we did
Sensor initialization changed the live simulation state	Used a copied MuJoCo data object during sensor setup; zero-residual regression tests then matched the nominal-only rollout.
Nominal gripper release can occur without a residual action	Agreed to attach a later release penalty to the preceding action. An initial release is reported as a zero-action failure and supplies no PPO sample.
Invalid sensor or inference states could otherwise enter training data	The smoke collector now aborts on invalid rollouts rather than updating from them; tests cover sensor faults and invalid episodes.
Force-on and force-off inputs needed to remain distinct	Added separate observation encoding and tests confirming force-off policies receive no wrench values.
The force penalty’s behavior needed checking	Tested its accounting through the collector with injected force samples. A separate force-stop diagnostic showed that the integral penalty can be small for a brief high-force spike; the reward weights are not yet validated.
Mock timings do not represent real model cost	Measured the original pi0.5 separately and ran a small combined integration test. These measurements still exclude a full PPO training workload.
The latest focused regression run passed 113 tests.


he latest focused regression run passed 113 tests.

Still unresolved
Alpine job 35998562 is queued for tomorrow; no fine-tuned checkpoint has been verified.
The PPO code is a smoke test, not a resumable multi-update trainer, and no residual policy has demonstrated insertion learning.
Training budgets, checkpoint cadence, evaluation procedure and reward adequacy still need validation.
The detailed methods and limitations are recorded in methods.md.