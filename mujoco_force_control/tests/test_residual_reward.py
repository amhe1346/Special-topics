import math

import pytest

from integrations.residual_reward import ForceSample, TaskState, interval_reward, potential
from probe_residual_reward import run_probe


def test_potential_and_progress_match_specification():
    initial = TaskState(0.00008, -0.01)
    final = TaskState(0, 0.020)
    assert potential(initial) == pytest.approx(0.5 * math.exp(-1))
    assert potential(final) == 1
    assert potential(TaskState(0, 0.050)) == 1
    result = interval_reward(initial, final, [], force_aware=False, termination="success")
    assert result.progress == pytest.approx(1 - 0.5 * math.exp(-1))
    assert result.terminal == 10
    assert result.total == pytest.approx(result.progress + 10)


def test_force_penalty_integrates_every_step_and_is_factorial():
    state = TaskState(0, 0)
    samples = [ForceSample(0.5, 0.002), ForceSample(1, 0.002), ForceSample(3, 0.002),
               ForceSample(6, 0.004)]
    off = interval_reward(state, state, samples, force_aware=False, termination="force_limit")
    on = interval_reward(state, state, samples, force_aware=True, termination="force_limit")
    assert on.force_excess_n_s == pytest.approx(0.024)
    assert on.force_penalty == pytest.approx(-0.024)
    assert off.force_excess_n_s == on.force_excess_n_s
    assert off.force_penalty == 0
    assert on.total == pytest.approx(-5.024)
    assert off.total == -5


@pytest.mark.parametrize("reason", ["dropped_peg", "force_limit", "workspace", "timeout", "release_requested"])
def test_failures_receive_terminal_penalty(reason):
    state = TaskState(0, 0)
    assert interval_reward(state, state, [], force_aware=False, termination=reason).terminal == -5


@pytest.mark.parametrize("reason", ["invalid_state", "wrench_input_error", "stale_observation",
                                     "inference_timeout", "unknown"])
def test_invalid_or_unknown_termination_is_not_scored(reason):
    with pytest.raises(ValueError):
        interval_reward(TaskState(0, 0), TaskState(0, 0), [], force_aware=True, termination=reason)


@pytest.mark.parametrize("state", [(-1, 0), (float("nan"), 0), (0, float("inf"))])
def test_invalid_geometry_is_rejected(state):
    with pytest.raises(ValueError):
        TaskState(*state)


@pytest.mark.parametrize("sample", [(-1, 1), (float("nan"), 1), (1, 0), (1, float("inf"))])
def test_invalid_force_sample_is_rejected(sample):
    with pytest.raises(ValueError):
        ForceSample(*sample)


@pytest.mark.parametrize(("timeout_s", "reason"), [(12.0, "success"), (1.0, "timeout")])
def test_recorded_classical_trajectory_is_scored_once_at_termination(timeout_s, reason):
    report = run_probe(0, timeout_s=timeout_s)
    assert report["outcome"]["reason"] == reason
    intervals = report["intervals"]
    assert len(intervals) > 1
    assert all(row["reward_off"]["terminal"] == 0 for row in intervals[:-1])
    assert intervals[-1]["reward_off"]["terminal"] == (10 if reason == "success" else -5)
    assert sum(row["duration_s"] for row in intervals) == pytest.approx(
        (len(report["physics_samples"]) - 1) * report["outcome"]["timestep_s"]
    )
    assert report["undiscounted_total_off"] == pytest.approx(report["undiscounted_total_on"])


def test_force_stop_trajectory_scores_only_until_termination():
    report = run_probe(2, lateral_range_m=0.005, tilt_range_deg=0.1)
    assert report["outcome"]["reason"] == "force_limit"
    assert report["outcome"]["peak_force_n"] > 5
    assert report["physics_samples"][-1]["contact_force_n"] > 5
    assert report["intervals"][-1]["reward_on"]["terminal"] == -5
    assert report["force_excess_n_s"] > 0
    assert report["undiscounted_total_off"] - report["undiscounted_total_on"] == pytest.approx(
        report["force_excess_n_s"]
    )
    assert len(report["physics_samples"]) == round(
        report["outcome"]["sim_time_s"] / report["outcome"]["timestep_s"]
    )
