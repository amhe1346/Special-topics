"""Define Stage D factorial variants; training is not implemented."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json

from integrations.residual_adapter import ResidualAdapter, ResidualPolicy, WrenchInput


class Variant(str, Enum):
    BASELINE = "baseline"
    OBSERVATION_ONLY = "observation_only"
    REWARD_ONLY = "reward_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class ExperimentConfig:
    variant: Variant
    force_observation: bool
    force_aware_reward: bool

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Variant):
            raise ValueError("variant must be a Variant")
        expected = {
            Variant.BASELINE: (False, False),
            Variant.OBSERVATION_ONLY: (True, False),
            Variant.REWARD_ONLY: (False, True),
            Variant.COMBINED: (True, True),
        }[self.variant]
        if (type(self.force_observation) is not bool or type(self.force_aware_reward) is not bool
                or (self.force_observation, self.force_aware_reward) != expected):
            raise ValueError("variant and experimental factors do not match")

    def make_adapter(
        self, policy: ResidualPolicy, wrench: WrenchInput, *,
        start_time_s: float, physics_timestep_s: float,
    ) -> ResidualAdapter:
        """Create a mock-testable adapter with this variant's observation factor."""
        return ResidualAdapter(
            policy, wrench, force_observation=self.force_observation,
            start_time_s=start_time_s, physics_timestep_s=physics_timestep_s,
        )


CONFIGURATIONS = (
    ExperimentConfig(Variant.BASELINE, False, False),
    ExperimentConfig(Variant.OBSERVATION_ONLY, True, False),
    ExperimentConfig(Variant.REWARD_ONLY, False, True),
    ExperimentConfig(Variant.COMBINED, True, True),
)


def get_config(variant: Variant) -> ExperimentConfig:
    if not isinstance(variant, Variant):
        raise ValueError("variant must be a Variant")
    return next(config for config in CONFIGURATIONS if config.variant == variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[variant.value for variant in Variant])
    args = parser.parse_args()
    configs = (get_config(Variant(args.variant)),) if args.variant else CONFIGURATIONS
    print(json.dumps({
        "schema": "stage-d-residual-configurations-v1",
        "status": "design_only",
        "training_implemented": False,
        "reward_implemented": True,
        "reward_scope": "simulation_rollout_pilot",
        "configurations": [asdict(config) for config in configs],
    }, indent=2))


if __name__ == "__main__":
    main()
