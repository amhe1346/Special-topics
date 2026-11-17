"""Probe a wall-paced, asynchronous policy loop in simulation only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controllers.pd_position_controller import PDPositionController
from envs.peg_episode import EpisodeConfig, PegEpisode
from evaluate_pi05 import PROMPT, _new_environment, _symmetric_range
from integrations.pi05_droid import infer_preview
from integrations.rollout import RolloutConfig
from integrations.replay_provenance import replay_provenance
from integrations.sim_observation import capture_observation
from run_pi05_preview import LocalOpenPiPolicy


def _infer_timed(policy, observation):
    start = time.monotonic()
    actions = infer_preview(policy, observation)
    return actions, start, time.monotonic()


def probe(env, renderer, policy, config: RolloutConfig) -> dict:
    """Hold the last target during inference; deliver chunks at control ticks."""
    controller = PDPositionController(env)
    target, _ = controller.read_joint_state()
    for _ in range(250):
        controller.step(target)
        env.step()
    mujoco.mj_forward(env.model, env.data)
    episode = PegEpisode(env, EpisodeConfig(timeout_s=config.timeout_s))
    if not episode._touching_finger():
        raise RuntimeError("Initial peg grasp is not established")
    dt = float(env.model.opt.timestep)
    if config.frequency_hz * dt > 1:
        raise ValueError("control frequency cannot exceed physics frequency")
    bounds = env.model.jnt_range[controller.joint_ids]
    period = 1.0 / config.frequency_hz
    pending = None
    chunk = ()
    chunk_index = 0
    deliveries = []
    inquiries = []
    step_lateness = []
    reason = None
    start = time.monotonic()
    next_tick = 0
    steps = 0
    loop_finished = start
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while reason is None:
                scheduled = start + steps * dt
                delay = scheduled - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                step_lateness.append(max(0.0, time.monotonic() - scheduled))
                elapsed_sim = float(env.data.time - episode.start_time)
                if elapsed_sim + dt / 2 >= next_tick * period:
                    next_tick += 1
                    if pending is not None and pending.done():
                        actions, infer_start, infer_end = pending.result()
                        if infer_end - infer_start > config.max_inference_s:
                            reason = "inference_timeout"
                            break
                        chunk = actions[:config.open_loop_horizon]
                        chunk_index = 0
                        inquiries[-1].update({
                            "inference_wall_s": infer_end - infer_start,
                            "response_wall_s": infer_end - start,
                            "response_observed_sim_time_s": float(env.data.time),
                        })
                        pending = None
                    if pending is not None and time.monotonic() - inquiries[-1]["submitted_wall_absolute_s"] > config.max_inference_s:
                        reason = "inference_timeout"
                        break
                    if chunk_index < len(chunk):
                        action = chunk[chunk_index]
                        chunk_index += 1
                        if not action.gripper_closed:
                            reason = "release_requested"
                            break
                        joints, _ = controller.read_joint_state()
                        delta = np.clip(
                            action.joint_delta_rad,
                            -config.max_joint_speed_rad_s / config.frequency_hz,
                            config.max_joint_speed_rad_s / config.frequency_hz,
                        )
                        target = np.clip(joints + delta, bounds[:, 0], bounds[:, 1])
                        deliveries.append({
                            "wall_s": time.monotonic() - start,
                            "sim_time_s": float(env.data.time),
                            "inquiry": len(inquiries) - 1,
                            "action_index": chunk_index - 1,
                            "joint_target_rad": target.tolist(),
                            "observation_age_sim_s": float(env.data.time) - inquiries[-1]["observation_time_s"],
                            "observation_age_wall_s": time.monotonic() - inquiries[-1]["observed_wall_absolute_s"],
                        })
                    elif pending is None:
                        observed = time.monotonic()
                        observation = capture_observation(env, renderer, prompt=PROMPT)
                        submitted = time.monotonic()
                        inquiries.append({
                            "observation_time_s": observation.time_s,
                            "observation_wall_s": observed - start,
                            "capture_wall_s": submitted - observed,
                            "observed_wall_absolute_s": observed,
                            "submitted_wall_absolute_s": submitted,
                        })
                        pending = executor.submit(_infer_timed, policy, observation)
                controller.step(target)
                reason = episode.step()
                steps += 1
                if reason is None and config.no_contact and episode.samples[-1].force_n > 0:
                    reason = "contact"
            loop_finished = time.monotonic()
            if pending is not None:
                _, infer_start, infer_end = pending.result()
                inquiries[-1].update({
                    "inference_wall_s": infer_end - infer_start,
                    "response_wall_s": infer_end - start,
                    "response_observed_sim_time_s": None,
                })
    finally:
        env.data.ctrl[controller.actuator_ids] = 0.0
    if reason in ("inference_timeout", "release_requested", "contact"):
        episode._finish_interval(float(env.data.time))
    for inquiry in inquiries:
        del inquiry["submitted_wall_absolute_s"]
        del inquiry["observed_wall_absolute_s"]
    wall_s = loop_finished - start
    intervals = np.diff([delivery["wall_s"] for delivery in deliveries])
    interval_tolerance = period + dt
    return {
        "reason": reason,
        "wall_s": wall_s,
        "worker_cleanup_wall_s": time.monotonic() - loop_finished,
        "sim_s": float(env.data.time - episode.start_time),
        "physics_steps": steps,
        "physics_step_lateness_p95_s": float(np.percentile(step_lateness, 95)),
        "physics_steps_late_over_one_timestep": int(np.count_nonzero(np.asarray(step_lateness) > dt)),
        "control_period_s": period,
        "action_intervals_wall_s": intervals.tolist(),
        "action_intervals_over_period": int(np.count_nonzero(intervals > period)),
        "action_interval_quantization_tolerance_s": interval_tolerance,
        "action_intervals_over_quantized_period": int(np.count_nonzero(intervals > interval_tolerance)),
        "action_interval_p95_s": float(np.percentile(intervals, 95)) if len(intervals) else None,
        "action_interval_max_s": float(np.max(intervals)) if len(intervals) else None,
        "peak_force_n": episode.peak_force_n,
        "inquiries": inquiries,
        "deliveries": deliveries,
        "limitations": [
            "Simulation only, not a hardware safety or real-time guarantee.",
            "Rendering runs on the simulator thread and may pause physics; scheduling lateness is reported.",
            "The previous position target is held while inference runs; returned actions use potentially stale observations.",
            "Inference is requested after a chunk finishes, not pipelined during the previous chunk.",
            "The 2 ms physics timestep quantizes nominal 15 Hz delivery intervals; wall-clock jitter also affects counts.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("results/stage_c_oracle_v2/manifest.json"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.source_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "stage-c-oracle-v2" or manifest.get("status") != "complete":
        parser.error("expected a complete stage-c-oracle-v2 manifest")
    if args.seed not in manifest["splits"]["validation"]:
        parser.error("timing probe is restricted to validation seeds")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.output.exists():
        parser.error("output already exists")
    randomization = manifest["randomization"]
    env = _new_environment(
        args.seed,
        _symmetric_range(randomization["lateral_offset_range_m"], "lateral offset"),
        _symmetric_range(randomization["fixture_tilt_xy_range_deg"], "fixture tilt"),
    )
    policy = LocalOpenPiPolicy(args.port, response_timeout=180.0)
    try:
        if policy.metadata.get("evaluation_server") != "stage-c-policy-server-v1":
            raise ValueError("server must be started by run_openpi_local_server.py")
        with mujoco.Renderer(env.model, height=224, width=224) as renderer:
            warmup_start = time.monotonic()
            infer_preview(policy, capture_observation(env, renderer, prompt=PROMPT))
            warmup_s = time.monotonic() - warmup_start
            policy.response_timeout = 5.0
            config = RolloutConfig(timeout_s=12.0, no_contact=False)
            result = probe(env, renderer, policy, config)
        report = {"schema": "stage-c-realtime-probe-v1", "seed": args.seed,
                  "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "policy_metadata": policy.metadata, "replay_provenance": replay_provenance(),
                  "rollout_config": asdict(config), "warmup_wall_s": warmup_s, **result}
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps({key: report[key] for key in ("reason", "wall_s", "sim_s", "physics_steps_late_over_one_timestep")}))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
