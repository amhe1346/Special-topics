"""Small, simulation-only PPO plumbing test; not a production training recipe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from integrations.residual_adapter import ResidualObservation
from integrations.residual_trajectory import episode_rewards, observation_vector


@dataclass(frozen=True)
class Sample:
    observation: np.ndarray
    latent: np.ndarray
    log_probability: float
    value: float


class SmokePolicy(nn.Module):
    """Sample latent Gaussian actions and map them into the adapter bounds."""

    def __init__(self, *, force_observation: bool):
        super().__init__()
        self.force_observation = force_observation
        features = 28 if force_observation else 22
        self.actor = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 7))
        self.critic = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_std = nn.Parameter(torch.full((7,), math.log(0.5)))
        self.samples: list[Sample] = []

    def infer(self, observation: ResidualObservation) -> np.ndarray:
        encoded = observation_vector(observation, force_observation=self.force_observation)
        tensor = torch.as_tensor(encoded, dtype=torch.float32)
        with torch.no_grad():
            distribution = Normal(self.actor(tensor), self.log_std.exp())
            latent = distribution.sample()
            log_probability = distribution.log_prob(latent).sum()
            value = self.critic(tensor).squeeze()
            correction = 0.001 * torch.tanh(latent)
        if not (torch.isfinite(latent).all() and torch.isfinite(log_probability)
                and torch.isfinite(value) and torch.isfinite(correction).all()):
            raise RuntimeError("non-finite policy inference; abort training")
        self.samples.append(Sample(encoded.copy(), latent.numpy().copy(),
                                   float(log_probability), float(value)))
        return correction.numpy()


def advantages_and_returns(rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GAE with gamma=.99, lambda=.95, and zero terminal bootstrap."""
    if rewards.ndim != 1 or values.shape != rewards.shape or not len(rewards):
        raise ValueError("rewards and values must be nonempty matching vectors")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("rewards and values must be finite")
    advantages = np.empty_like(rewards, dtype=np.float64)
    carry = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + 0.99 * next_value - values[index]
        carry = delta + 0.99 * 0.95 * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, advantages + values


def update(policy: SmokePolicy, episodes: list[tuple[list[Sample], dict]]) -> dict:
    """Validate all episodes before a two-epoch full-batch clipped PPO update."""
    samples: list[Sample] = []
    advantages_list = []
    returns_list = []
    outcomes = []
    for episode_samples, result in episodes:
        mapped = episode_rewards(result, len(episode_samples))
        outcomes.append({
            "reason": mapped.terminal_reason, "actions": len(episode_samples),
            "unassigned_release_reward": mapped.unassigned_release_reward,
            "undiscounted_reward_total": result["residual_experiment"]["undiscounted_reward_total"],
            "sim_time_s": result.get("sim_time_s"),
            "physics_steps": result.get("steps"),
            "nominal_inference_timing": result.get("inference_timing"),
            "force_excess_n_s": sum(item["breakdown"]["force_excess_n_s"]
                                    for item in result["residual_experiment"]["reward_intervals"]),
            "force_penalty": sum(item["breakdown"]["force_penalty"]
                                 for item in result["residual_experiment"]["reward_intervals"]),
        })
        if episode_samples:
            values = np.asarray([sample.value for sample in episode_samples], dtype=np.float64)
            advantages, returns = advantages_and_returns(mapped.rewards, values)
            advantages_list.append(advantages)
            returns_list.append(returns)
            samples.extend(episode_samples)
    if not samples:
        raise RuntimeError("no residual actions collected; cannot update PPO")

    observations = torch.as_tensor(np.stack([sample.observation for sample in samples]), dtype=torch.float32)
    latents = torch.as_tensor(np.stack([sample.latent for sample in samples]), dtype=torch.float32)
    old_log_prob = torch.tensor([sample.log_probability for sample in samples], dtype=torch.float32)
    advantages = torch.as_tensor(np.concatenate(advantages_list), dtype=torch.float32)
    returns = torch.as_tensor(np.concatenate(returns_list), dtype=torch.float32)
    if advantages.numel() > 1:
        std = advantages.std(unbiased=False)
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / std
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    initial_actor = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    losses = []
    for _ in range(2):
        distribution = Normal(policy.actor(observations), policy.log_std.exp())
        log_prob = distribution.log_prob(latents).sum(dim=-1)
        ratio = (log_prob - old_log_prob).exp()
        objective = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
        policy_loss = -objective.mean()
        value_loss = (policy.critic(observations).squeeze(-1) - returns).square().mean()
        loss = policy_loss + 0.5 * value_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite PPO loss; abort training")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in policy.parameters()):
            raise RuntimeError("non-finite PPO parameters; abort training")
        losses.append({"policy": float(policy_loss.detach()), "value": float(value_loss.detach())})
    change = math.sqrt(sum(float((parameter.detach() - initial).square().sum())
                           for parameter, initial in zip(policy.parameters(), initial_parameters)))
    actor_change = math.sqrt(sum(float((parameter.detach() - initial).square().sum())
                                 for parameter, initial in zip(policy.actor.parameters(), initial_actor)))
    return {
        "action_samples": len(samples), "episodes": outcomes, "losses": losses,
        "parameter_delta_l2": change, "actor_parameter_delta_l2": actor_change,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
        "epochs": 2, "learning_rate": 3e-4,
    }
