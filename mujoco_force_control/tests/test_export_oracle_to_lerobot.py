import numpy as np
import pytest

from export_oracle_to_lerobot import _validate_timing


def test_source_timestamp_jitter_is_allowed_when_mean_rate_is_15_hz():
    times = np.array([0.0, 0.066, 0.134, 0.2])
    _validate_timing(times, "trial.npz")


@pytest.mark.parametrize("times", [
    np.array([0.0, 0.06, 0.12]),
    np.array([0.0, 0.068, 0.136]),
    np.array([0.0, 0.066, 0.066]),
])
def test_invalid_source_timestamp_cadence_is_rejected(times):
    with pytest.raises(ValueError):
        _validate_timing(times, "trial.npz")
