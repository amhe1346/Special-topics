"""Language label generation for training data."""
import numpy as np


class LanguageSampler:
    """Generate simple language labels for trajectories."""

    LABELS = {
        'sinusoid': [
            "oscillate smoothly",
            "move in a wave pattern",
            "sinusoidal motion",
        ],
        'step': [
            "move to home position",
            "reach the target pose",
            "go to home",
        ],
        'random_walk': [
            "explore the workspace",
            "move around freely",
            "random trajectory",
        ],
    }

    @staticmethod
    def sample(scenario):
        """Get a random language label for a scenario.

        Args:
            scenario: One of ['sinusoid', 'step', 'random_walk']

        Returns:
            Language string
        """
        if scenario not in LanguageSampler.LABELS:
            scenario = 'random_walk'
        return np.random.choice(LanguageSampler.LABELS[scenario])
