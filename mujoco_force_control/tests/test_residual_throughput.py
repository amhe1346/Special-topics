import json
import sys
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_residual_throughput as throughput
from integrations.residual_experiments import Variant


class Renderer:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


def test_benchmark_measures_all_variants_and_excludes_warmup(monkeypatch):
    monkeypatch.setattr(throughput.mujoco, "Renderer", Renderer)
    report = throughput.benchmark(repeats=1, timeout_s=0.1)
    assert report["status"] == "complete"
    assert report["warmup_episodes_per_variant"] == 1
    assert len(report["warmups"]) == 4
    assert len(report["measurements"]) == 4
    assert {row["variant"] for row in report["measurements"]} == {v.value for v in Variant}
    for row, summary in zip(report["measurements"], report["summaries"], strict=True):
        assert row["reason"] == "timeout"
        assert row["physics_steps"] == 50
        assert row["action_steps"] == row["residual_calls"] == 2
        assert row["end_to_end_wall_s"] >= row["rollout_wall_s"] > 0
        assert row["end_to_end_wall_s"] == pytest.approx(sum(
            row[key] for key in (
                "environment_setup_wall_s", "renderer_setup_wall_s",
                "rollout_wall_s", "renderer_teardown_wall_s",
            )
        ))
        assert summary["action_steps_per_wall_s"] == pytest.approx(
            row["action_steps"] / row["end_to_end_wall_s"]
        )


def test_benchmark_rejects_bad_arguments():
    with pytest.raises(ValueError, match="repeats"):
        throughput.benchmark(0, 1)
    with pytest.raises(ValueError, match="timeout"):
        throughput.benchmark(1, 0)


def test_timing_and_aggregation_with_deterministic_clock(monkeypatch):
    class Environment:
        model = None

        def reset(self, **kwargs):
            pass

    ticks = count()
    monkeypatch.setattr(throughput, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(throughput, "PegInHoleEnv", Environment)
    monkeypatch.setattr(throughput.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(throughput, "run_rollout", lambda *args, **kwargs: {
        "residual_experiment": {"reward_status": "complete"},
        "reason": "timeout", "sim_time_s": 1.0, "steps": 500,
        "action_steps": 15, "inference_timing": {"calls": 2},
    })
    report = throughput.benchmark(2, 1)
    for row in report["measurements"]:
        assert row["environment_setup_wall_s"] == 1
        assert row["renderer_setup_wall_s"] == 1
        assert row["rollout_wall_s"] == 1
        assert row["renderer_teardown_wall_s"] == 1
        assert row["end_to_end_wall_s"] == 4
    for summary in report["summaries"]:
        assert summary["episodes"] == 2
        assert summary["end_to_end_wall_s"] == 8
        assert summary["action_steps_per_wall_s"] == 30 / 8
        assert summary["physics_steps_per_wall_s"] == 1000 / 8


def test_invalid_rollout_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(throughput.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(throughput, "run_rollout", lambda *args, **kwargs: {
        "residual_experiment": {"reward_status": "invalid"}, "reason": "sensor_error",
    })
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(RuntimeError, match="sensor_error"):
        throughput.main()
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "error"
    assert "sensor_error" in report["error"]["message"]


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    report = output / "report.json"
    report.write_text("existing")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(FileExistsError):
        throughput.main()
    assert report.read_text() == "existing"
