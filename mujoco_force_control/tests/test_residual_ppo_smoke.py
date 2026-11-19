import sys
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import run_residual_ppo_smoke as smoke
from integrations.residual_adapter import ResidualObservation
from envs.peg_episode import PegEpisode
from integrations.residual_experiments import Variant
from integrations.residual_ppo_smoke import SmokePolicy, advantages_and_returns, update
from integrations.rollout import RolloutConfig
from sensors.residual_wrench import ResidualWrenchInput, WrenchInputError
from tests.test_residual_rollout import Renderer


@pytest.mark.parametrize("variant", list(Variant))
def test_real_rollout_ppo_update(monkeypatch, variant):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    report = smoke.run(variant, 0, 2, RolloutConfig(timeout_s=0.1, no_contact=False))
    result = report["update"]
    assert result["action_samples"] == 4
    assert [episode["reason"] for episode in result["episodes"]] == ["timeout", "timeout"]
    assert result["parameter_delta_l2"] > 0
    assert result["actor_parameter_delta_l2"] > 0
    assert all(np.isfinite(item["policy"]) and np.isfinite(item["value"]) for item in result["losses"])


def test_gae_terminal_bootstrap():
    advantages, returns = advantages_and_returns(np.array([1.0, -5.0]), np.array([0.5, 0.25]))
    assert advantages[1] == pytest.approx(-5.25)
    assert advantages[0] == pytest.approx(1 + 0.99 * 0.25 - 0.5 + 0.99 * 0.95 * -5.25)
    np.testing.assert_allclose(returns, advantages + [0.5, 0.25])


def test_actions_bounded_and_latent_log_probability_matches():
    torch.manual_seed(0)
    policy = SmokePolicy(force_observation=False)
    obs = ResidualObservation(0, np.zeros(7), np.zeros(7), 1, np.zeros(7), True)
    action = policy.infer(obs)
    assert np.all(np.abs(action) <= 0.001)
    sample = policy.samples[0]
    with torch.no_grad():
        distribution = torch.distributions.Normal(
            policy.actor(torch.tensor(sample.observation, dtype=torch.float32)), policy.log_std.exp()
        )
        expected = distribution.log_prob(torch.tensor(sample.latent)).sum()
    assert sample.log_probability == pytest.approx(float(expected))


def test_invalid_episode_aborts_before_update_and_further_episodes(monkeypatch):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    calls = []

    def invalid(*args, **kwargs):
        calls.append(1)
        return {"reason": "wrench_input_error", "residual_experiment": {"reward_status": "invalid"}}

    monkeypatch.setattr(smoke, "run_rollout", invalid)
    with pytest.raises(RuntimeError, match="abort training"):
        smoke.run(Variant.BASELINE, 0, 3, RolloutConfig(timeout_s=0.1, no_contact=False))
    assert len(calls) == 1


def test_zero_action_release_never_causes_update():
    policy = SmokePolicy(force_observation=False)
    result = {
        "reason": "release_requested", "action_steps": 0,
        "residual_experiment": {
            "reward_status": "complete", "undiscounted_reward_total": -5,
            "reward_intervals": [{"kind": "nominal_release_event", "termination": "release_requested",
                                  "breakdown": {"total": -5, "force_excess_n_s": 0, "force_penalty": 0}}],
        },
    }
    initial = [parameter.detach().clone() for parameter in policy.parameters()]
    with pytest.raises(RuntimeError, match="no residual actions"):
        update(policy, [([], result)])
    assert all(torch.equal(before, after) for before, after in zip(initial, policy.parameters()))


@pytest.mark.parametrize("variant", list(Variant))
def test_force_penalty_reaches_ppo_update(monkeypatch, variant):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    original_step = PegEpisode.step

    def force_stop(episode):
        original_step(episode)
        episode.samples[-1] = replace(episode.samples[-1], force_n=6.0)
        episode.peak_force_n = 6.0
        episode.reason = "force_limit"
        return "force_limit"

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    monkeypatch.setattr(PegEpisode, "step", force_stop)
    report = smoke.run(variant, 0, 2, RolloutConfig(timeout_s=0.1, no_contact=False))
    for episode in report["update"]["episodes"]:
        assert episode["reason"] == "force_limit"
        assert episode["actions"] == 1
        assert episode["force_excess_n_s"] == pytest.approx(0.01)
        assert episode["force_penalty"] == pytest.approx(
            -0.01 if variant in (Variant.REWARD_ONLY, Variant.COMBINED) else 0
        )


def test_release_after_action_reaches_update(monkeypatch):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Release:
        def infer(self, observation):
            actions = np.zeros((15, 8))
            actions[0, 7] = 1
            return {"actions": actions}

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    report = smoke.run(
        Variant.BASELINE, 0, 2, RolloutConfig(timeout_s=0.3, no_contact=False),
        nominal=Release(), nominal_metadata={"kind": "release_test"},
    )
    assert report["nominal_policy"]["kind"] == "release_test"
    assert report["update"]["action_samples"] == 2
    for episode in report["update"]["episodes"]:
        assert episode["reason"] == "release_requested"
        assert episode["unassigned_release_reward"] == 0
        assert episode["undiscounted_reward_total"] < -4


def test_sensor_fault_mid_action_aborts_without_update(monkeypatch):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    original = ResidualWrenchInput.update

    def fault(sensor, time_s):
        if time_s >= 0.58:
            raise WrenchInputError("mid-episode sensor fault")
        original(sensor, time_s)

    def unexpected_update(*args):
        pytest.fail("invalid rollout reached optimizer")

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    monkeypatch.setattr(ResidualWrenchInput, "update", fault)
    monkeypatch.setattr(smoke, "update", unexpected_update)
    with pytest.raises(RuntimeError, match="wrench_input_error: mid-episode sensor fault"):
        smoke.run(Variant.BASELINE, 0, 2, RolloutConfig(timeout_s=0.3, no_contact=False))


def test_initial_release_reported_without_sample_in_mixed_batch(monkeypatch):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Mixed:
        def __init__(self):
            self.calls = 0

        def infer(self, observation):
            self.calls += 1
            actions = np.zeros((15, 8))
            if self.calls > 1:
                actions[:, 7] = 1
            return {"actions": actions}

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    report = smoke.run(
        Variant.BASELINE, 0, 2, RolloutConfig(timeout_s=0.1, no_contact=False),
        nominal=Mixed(), nominal_metadata={"kind": "mixed_test"},
    )
    episodes = report["update"]["episodes"]
    assert episodes[0]["reason"] == "release_requested"
    assert episodes[0]["actions"] == 0
    assert episodes[0]["unassigned_release_reward"] == -5
    assert episodes[1]["reason"] == "timeout"
    assert report["update"]["action_samples"] == episodes[1]["actions"] == 2


def test_invalid_second_episode_aborts_before_update(monkeypatch):
    class ContextRenderer(Renderer):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    original = smoke.run_rollout
    calls = []

    def second_invalid(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            return {"reason": "inference_timeout", "residual_experiment": {"reward_status": "invalid"}}
        return original(*args, **kwargs)

    def unexpected_update(*args):
        pytest.fail("invalid batch reached optimizer")

    monkeypatch.setattr(smoke.mujoco, "Renderer", ContextRenderer)
    monkeypatch.setattr(smoke, "run_rollout", second_invalid)
    monkeypatch.setattr(smoke, "update", unexpected_update)
    with pytest.raises(RuntimeError, match="inference_timeout"):
        smoke.run(Variant.BASELINE, 0, 3, RolloutConfig(timeout_s=0.1, no_contact=False))
    assert len(calls) == 2


def test_cli_refuses_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    report = output / "report.json"
    report.write_text("unchanged")
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    with pytest.raises(FileExistsError):
        smoke.main()
    assert report.read_text() == "unchanged"
