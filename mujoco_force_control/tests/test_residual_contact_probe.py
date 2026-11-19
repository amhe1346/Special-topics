import numpy as np
import pytest

from integrations.residual_experiments import Variant
from probe_residual_contact import run_probe


def test_contact_probe_paired_observations_and_delayed_data():
    off, off_rows = run_probe(Variant.BASELINE, 0)
    on, on_rows = run_probe(Variant.OBSERVATION_ONLY, 0)
    assert off_rows == on_rows
    assert off["outcome"] == on["outcome"]
    assert off["peak_contact_force_n"] > 0.1
    assert off["contact_samples_above_threshold"] > 0
    assert off["shadow_policy_calls"] > 0
    assert off["shadow_force_observation_calls"] == 0
    assert on["shadow_force_observation_calls"] == on["shadow_policy_calls"] == off["shadow_policy_calls"]
    assert off["sample_delay_min_s"] == pytest.approx(2 * off["timestep_s"])
    assert off["sample_delay_max_s"] == pytest.approx(2 * off["timestep_s"])
    assert all(np.isfinite(row["contact_force_n"]) for row in off_rows)
