"""Robot-driven Stage B insertion trials with fixed fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
from collections.abc import Callable
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np

from controllers.force_controller import EndEffectorForceController
from controllers.pd_position_controller import PDPositionController
from controllers.phase4_controllers import (
    CartesianImpedanceController,
    HybridPositionForceController,
    _rotation_error,
)
from envs.peg_episode import EpisodeConfig, PegEpisode
from envs.peg_in_hole import PegInHoleEnv
from sensors.load_cell import EndEffectorLoadCell
from sensors.noisy_wrench import NoisyWrench


IK_GAIN = 0.005


def classify_failure(
    *, depth: float, lateral_error: float, surface_contact: bool, radial_distance: float,
    surface_seconds: float, max_above_s: float, lateral_exceeded: bool,
) -> tuple[str, str]:
    """Classify surface failures conservatively; leave ambiguous positions unclassified."""
    if surface_contact and depth <= 0.001 and lateral_error <= 0.00008:
        geometric = "entry_failure"
    elif surface_contact and depth <= 0.001:
        geometric = "surface_miss"
    elif radial_distance > 0.018:
        geometric = "drift_off"
    else:
        geometric = "unclassified"
    force = (
        "excessive_pressing" if max_above_s + 1e-12 >= 1.0
        else "unstable_contact" if surface_seconds < 1.0 or lateral_exceeded
        else "stalled"
    )
    return geometric, force


class DelayedLoadCell(EndEffectorLoadCell):
    """Provide an explicitly updated delayed wrench to the hybrid controller."""

    def __init__(self, env: PegInHoleEnv) -> None:
        super().__init__(env)
        self.observation = super().read_wrench()

    def read_wrench(self) -> np.ndarray:
        return self.observation.copy()


@dataclass(frozen=True)
class TrialConfig:
    seed: int = 0
    lateral_range_m: float = 0.00005
    tilt_range_deg: float = 0.03
    speed_m_s: float = 0.005
    timeout_s: float = 12.0
    noise_force_n: float = 0.05
    noise_torque_nm: float = 0.005
    latency_steps: int = 2

    def __post_init__(self) -> None:
        values = (self.lateral_range_m, self.tilt_range_deg, self.speed_m_s, self.timeout_s, self.noise_force_n, self.noise_torque_nm)
        if not all(np.isfinite(v) for v in values) or min(values) < 0 or self.speed_m_s == 0 or self.timeout_s == 0:
            raise ValueError("trial configuration must be finite and non-negative, with positive speed and timeout")
        if not isinstance(self.latency_steps, int) or self.latency_steps < 0:
            raise ValueError("latency_steps must be a non-negative integer")
        if not isinstance(self.seed, int) or self.seed < 0 or self.tilt_range_deg >= 90:
            raise ValueError("seed must be non-negative and tilt_range_deg must be less than 90")


def run_trial(
    method: str,
    config: TrialConfig,
    *,
    record_trace: bool = False,
    on_step: Callable[[PegInHoleEnv], None] | None = None,
) -> dict:
    if method not in ("position", "hybrid", "impedance"):
        raise ValueError("method must be position, hybrid, or impedance")
    env = PegInHoleEnv()
    env.reset(seed=config.seed, lateral_range=config.lateral_range_m)
    random = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    tilt = random.uniform(-config.tilt_range_deg, config.tilt_range_deg, 2)
    angle = float(np.deg2rad(np.linalg.norm(tilt)))
    if angle:
        quat = np.empty(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([tilt[0], tilt[1], 0.0]) / np.linalg.norm(tilt), angle)
        env.data.mocap_quat[env.fixture_mocap_id] = quat
        mujoco.mj_forward(env.model, env.data)
    joint = PDPositionController(env)
    target_joint, _ = joint.read_joint_state()
    for _ in range(250):
        joint.step(target_joint)
        env.step()
    mujoco.mj_forward(env.model, env.data)
    if (abs(float(env.data.time) - 250 * env.model.opt.timestep) > 1e-8
            or any(warning.number for warning in env.data.warning)
            or not all(np.all(np.isfinite(value)) for value in (env.data.qpos, env.data.qvel, env.data.sensordata))):
        raise RuntimeError("Invalid simulation state while settling the initial grasp")
    delayed_cell = DelayedLoadCell(env)
    force_control = EndEffectorForceController(joint, load_cell=delayed_cell, kp=0.05)
    impedance = CartesianImpedanceController(force_control)
    hybrid = HybridPositionForceController(joint, force_control)
    load_cell = EndEffectorLoadCell(env)
    noise = NoisyWrench(
        seed=int(np.random.SeedSequence([config.seed, 2]).generate_state(1)[0]),
        latency_steps=config.latency_steps,
        noise_std=(config.noise_force_n,) * 3 + (config.noise_torque_nm,) * 3,
    )
    episode = PegEpisode(env, EpisodeConfig(timeout_s=config.timeout_s))
    touching = set()
    for contact in env.data.contact:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if contact.dist > 0:
            continue
        if geom1 == episode.contact.peg_geom:
            touching.add(int(env.model.geom_bodyid[geom2]))
        elif geom2 == episode.contact.peg_geom:
            touching.add(int(env.model.geom_bodyid[geom1]))
    if not episode.finger_bodies <= touching:
        raise RuntimeError("Initial two-finger grasp failed after settling; trial not started")
    site = force_control.site_id
    start = env.data.site_xpos[site].copy()
    rotation = env.data.site_xmat[site].reshape(3, 3).copy()
    fixture = env.data.mocap_pos[env.fixture_mocap_id].copy()
    initial_tip = env.data.site_xpos[env.peg_tip_id].copy()
    nominal = start.copy()
    nominal[:2] += fixture[:2] - initial_tip[:2]
    target_joint = joint.read_joint_state()[0]
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    max_steps = int(np.ceil(config.timeout_s / env.model.opt.timestep))
    raw_baseline = load_cell.read_wrench()
    if not np.all(np.isfinite(raw_baseline)):
        raise RuntimeError("Initial load-cell tare is invalid")
    surface_seconds = 0.0
    surface_above_s = 0.0
    max_surface_above_s = 0.0
    lateral_exceeded = False
    trace = []
    for _ in range(max_steps):
        dt = env.model.opt.timestep
        measurement_time = float(env.data.time)
        raw = load_cell.read_wrench() - raw_baseline
        if not np.isfinite(measurement_time) or not np.all(np.isfinite(raw)):
            episode.reason = "invalid_state"
            end_time = measurement_time if np.isfinite(measurement_time) else (
                episode.samples[-1].time_s if episode.samples else episode.start_time
            )
            episode._finish_interval(end_time)
            break
        delayed = noise.update(measurement_time, raw)
        if delayed is not None:
            delayed_cell.observation = delayed[1] + raw_baseline
        nominal[2] = max(start[2] - 0.038, nominal[2] - config.speed_m_s * dt)
        if method in ("position", "hybrid"):
            mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site)
            jac = np.vstack((jacp[:, joint.dof_indices], jacr[:, joint.dof_indices]))
            error = np.r_[nominal - env.data.site_xpos[site],
                          _rotation_error(env.data.site_xmat[site].reshape(3, 3), rotation)]
            change = jac.T @ np.linalg.solve(jac @ jac.T + 0.0025 * np.eye(6), error)
            target_joint += np.clip(change * IK_GAIN, -0.002, 0.002)
            if method == "hybrid" and delayed is not None:
                hybrid.step(target_joint, raw_baseline)
            else:
                joint.step(target_joint)
        else:
            impedance.step(nominal, rotation)
        reason = episode.step()
        if on_step is not None:
            on_step(env)
        if reason == "invalid_state":
            break
        surface = episode.contact.measure(surface_only=True)
        frame = env.data.site_xmat[env.hole_site_id].reshape(3, 3)
        local_force = frame.T @ np.asarray(surface.world_vector)
        if surface.contact_count:
            surface_seconds += dt
            lateral_exceeded |= bool(np.linalg.norm(local_force[:2]) > abs(local_force[2]))
            surface_above_s = surface_above_s + dt if surface.magnitude_sum > episode.config.reference_force_n else 0.0
            max_surface_above_s = max(max_surface_above_s, surface_above_s)
        else:
            surface_above_s = 0.0
        if record_trace:
            sample = episode.samples[-1]
            trace.append({
                "time_s": sample.time_s, "depth_m": sample.depth_m,
                "contact_force_n": sample.force_n, "contact_force_world_n": sample.force_world_n,
                "surface_force_n": surface.magnitude_sum, "surface_contact_count": surface.contact_count,
                "raw_wrench_time_s": measurement_time, "raw_wrench_local": (raw + raw_baseline).tolist(),
                "observed_wrench_local": None if delayed is None else delayed_cell.observation.tolist(),
                "observation_time_s": None if delayed is None else delayed[0],
                "seated": sample.seated,
            })
        if reason is not None:
            break
    if episode.reason is None:
        raise RuntimeError("Trial loop ended without a termination reason")
    geometry = None if episode.reason == "invalid_state" else env.measure_geometry()
    geometric_mode = None
    force_mode = None
    if episode.reason not in ("success", "invalid_state"):
        assert geometry is not None
        geometric_mode, force_mode = classify_failure(
            depth=geometry.depth, lateral_error=geometry.lateral_error,
            surface_contact=bool(surface.contact_count),
            radial_distance=float(np.linalg.norm(env.data.site_xpos[env.peg_tip_id, :2] - fixture[:2])),
            surface_seconds=surface_seconds, max_above_s=max_surface_above_s,
            lateral_exceeded=lateral_exceeded,
        )
    result = {
        "method": method, "config": asdict(config), "reason": episode.reason,
        "sim_time_s": float(env.data.time - episode.start_time) if np.isfinite(env.data.time) else None,
        "final_depth_m": None if geometry is None else geometry.depth,
        "final_lateral_m": None if geometry is None else geometry.lateral_error,
        "final_angle_rad": None if geometry is None else geometry.angular_error, "peak_force_n": episode.peak_force_n,
        "above_reference_intervals": episode.above_reference_intervals,
        "fixture_position_m": fixture.tolist(), "fixture_tilt_xy_deg": tilt.tolist(),
        "final_fixture_position_m": (
            env.data.mocap_pos[env.fixture_mocap_id].tolist()
            if np.all(np.isfinite(env.data.mocap_pos[env.fixture_mocap_id])) else None
        ),
        "surface_contact_s": surface_seconds, "surface_max_above_reference_s": max_surface_above_s,
        "surface_lateral_exceeded_axial": lateral_exceeded,
        "geometric_failure_mode": geometric_mode, "force_failure_mode": force_mode,
        "timestep_s": float(env.model.opt.timestep), "mujoco_version": mujoco.__version__,
        "wrench_tare_local": raw_baseline.tolist(),
        "ik_gain": IK_GAIN,
    }
    if record_trace:
        result["trace"] = trace
    return result


class ViewerClosed(Exception):
    """The viewer was closed before a trial terminated."""


class TrialViewer:
    """Show the actual benchmark physics at approximately real-time speed."""

    def __init__(self) -> None:
        self.stack = ExitStack()
        self.viewer = None
        self.env: PegInHoleEnv | None = None
        self.viewer_thread: threading.Thread | None = None
        self.next_sync = 0.0
        self.start_wall = 0.0
        self.start_sim = 0.0

    def __enter__(self) -> TrialViewer:
        self.stack.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()
        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=10)
            if self.viewer_thread.is_alive():
                raise RuntimeError("MuJoCo viewer thread did not shut down")

    def __call__(self, env: PegInHoleEnv) -> None:
        if self.viewer is None:
            import mujoco.viewer

            self.env = env
            existing_threads = set(threading.enumerate())
            self.viewer = self.stack.enter_context(mujoco.viewer.launch_passive(env.model, env.data))
            if sys.platform != "darwin":
                new_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
                matching = [thread for thread in new_threads if "_launch_internal" in thread.name]
                if len(matching) == 1:
                    self.viewer_thread = matching[0]
                elif len(new_threads) == 1:
                    self.viewer_thread = new_threads[0]
                elif new_threads:
                    raise RuntimeError("Could not identify MuJoCo viewer thread for safe shutdown")
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = env.data.site_xpos[env.hole_site_id]
                self.viewer.cam.distance = 0.35
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -25
            self.start_wall = time.monotonic()
            self.start_sim = float(env.data.time)
        if not self.viewer.is_running():
            raise ViewerClosed
        now = time.monotonic()
        delay = self.start_wall + float(env.data.time) - self.start_sim - now
        if delay > 0:
            time.sleep(delay)
        if time.monotonic() >= self.next_sync:
            self.viewer.sync()
            self.next_sync = time.monotonic() + 1 / 30

    def wait_until_closed(self) -> None:
        if self.viewer is None:
            return
        self.viewer.sync()
        while self.viewer.is_running():
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("position", "hybrid", "impedance", "all"), default="all")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lateral-range-m", type=float, default=TrialConfig.lateral_range_m)
    parser.add_argument("--tilt-range-deg", type=float, default=TrialConfig.tilt_range_deg)
    parser.add_argument("--speed-m-s", type=float, default=TrialConfig.speed_m_s)
    parser.add_argument("--timeout-s", type=float, default=TrialConfig.timeout_s)
    parser.add_argument("--noise-force-n", type=float, default=TrialConfig.noise_force_n)
    parser.add_argument("--noise-torque-nm", type=float, default=TrialConfig.noise_torque_nm)
    parser.add_argument("--latency-steps", type=int, default=TrialConfig.latency_steps)
    parser.add_argument("--trace-dir", type=Path, help="directory for per-trial JSON traces (must not already exist)")
    parser.add_argument("--viewer", action="store_true", help="show each trial in the interactive MuJoCo viewer")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        config = TrialConfig(
            seed=args.seed, lateral_range_m=args.lateral_range_m, tilt_range_deg=args.tilt_range_deg,
            speed_m_s=args.speed_m_s, timeout_s=args.timeout_s, noise_force_n=args.noise_force_n,
            noise_torque_nm=args.noise_torque_nm, latency_steps=args.latency_steps,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
    methods = ("position", "hybrid", "impedance") if args.method == "all" else (args.method,)
    for seed in range(args.seed, args.seed + args.trials):
        for method in methods:
            trial_config = TrialConfig(**(asdict(config) | {"seed": seed}))
            if args.viewer:
                try:
                    with TrialViewer() as viewer:
                        result = run_trial(method, trial_config, record_trace=args.trace_dir is not None, on_step=viewer)
                        print(
                            f"{method} seed {seed}: {result['reason']}; "
                            f"peak force {result['peak_force_n']:.3f} N. Close the viewer to continue.",
                            file=sys.stderr, flush=True,
                        )
                        viewer.wait_until_closed()
                except ViewerClosed:
                    print("Viewer closed before trial completion; trial aborted.", file=sys.stderr)
                    raise SystemExit(1)
            else:
                result = run_trial(method, trial_config, record_trace=args.trace_dir is not None)
            if args.trace_dir is not None:
                trace = result.pop("trace")
                path = args.trace_dir / f"{method}-{seed}.json"
                with path.open("x", encoding="utf-8") as file:
                    json.dump({"summary": result, "samples": trace}, file, allow_nan=False)
                result["trace_path"] = str(path)
            print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
