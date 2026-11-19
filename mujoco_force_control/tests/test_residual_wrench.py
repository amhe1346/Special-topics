import mujoco
import numpy as np
import pytest

from envs.peg_in_hole import PegInHoleEnv
from sensors.load_cell import EndEffectorLoadCell
from sensors.noisy_wrench import NoisyWrench
from sensors.residual_wrench import ResidualWrenchInput, WrenchInputError


class Source:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def read_wrench(self):
        return self.value.copy()


def test_tare_seeded_noise_and_exact_two_step_delay():
    source = Source([1, 2, 3, 4, 5, 6])
    sensor = ResidualWrenchInput(source, trial_seed=7, timestep_s=0.002)
    seed = int(np.random.SeedSequence([7, 2]).generate_state(1)[0])
    expected = NoisyWrench(seed=seed, noise_std=(0.05,) * 3 + (0.005,) * 3, latency_steps=2)
    for step in range(5):
        source.value = np.arange(6.0) + step
        wanted = expected.update(step * 0.002, source.value - np.arange(1.0, 7.0))
        sensor.update(step * 0.002)
        if wanted is not None:
            result = sensor.sample(step * 0.002)
            assert result.time_s == pytest.approx((step - 2) * 0.002)
            np.testing.assert_array_equal(result.wrench_local, wanted[1])
    result.wrench_local[:] = 99
    assert not np.all(sensor.sample(0.008).wrench_local == 99)
    tare = sensor.tare_local
    tare[:] = 99
    np.testing.assert_array_equal(sensor.tare_local, np.arange(1.0, 7.0))


def make_ready():
    source = Source(np.zeros(6))
    sensor = ResidualWrenchInput(source, trial_seed=0, timestep_s=0.002)
    for step in range(3):
        sensor.update(step * 0.002)
    return source, sensor


def test_unavailable_stale_future_and_invalid_time_fault():
    sensor = ResidualWrenchInput(Source(np.zeros(6)), trial_seed=0, timestep_s=0.002)
    with pytest.raises(WrenchInputError, match="unavailable"):
        sensor.sample(0)
    with pytest.raises(WrenchInputError, match="faulted"):
        sensor.update(0)
    _, sensor = make_ready()
    assert sensor.sample(0.1).time_s == 0
    with pytest.raises(WrenchInputError, match="stale"):
        sensor.sample(0.1000000001)
    _, sensor = make_ready()
    with pytest.raises(WrenchInputError, match="future-dated"):
        sensor.sample(0.001)
    _, sensor = make_ready()
    with pytest.raises(WrenchInputError, match="invalid control timestamp"):
        sensor.sample(float("nan"))


def test_missing_steps_invalid_reading_and_invalid_tare_fail():
    source, sensor = make_ready()
    with pytest.raises(WrenchInputError, match="once per physics step"):
        sensor.update(0.008)
    with pytest.raises(WrenchInputError, match="faulted"):
        sensor.sample(0.004)
    source, sensor = make_ready()
    source.value[0] = np.nan
    with pytest.raises(WrenchInputError, match="invalid wrist wrench"):
        sensor.update(0.006)
    with pytest.raises(WrenchInputError, match="faulted"):
        sensor.sample(0.006)
    with pytest.raises(WrenchInputError, match="invalid wrist wrench"):
        ResidualWrenchInput(Source([1, 2]), trial_seed=0, timestep_s=0.002)


def test_real_mujoco_load_cell_produces_timestamped_delayed_wrench():
    env = PegInHoleEnv()
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    source = EndEffectorLoadCell(env)
    sensor = ResidualWrenchInput(source, trial_seed=0, timestep_s=env.model.opt.timestep)
    start = float(env.data.time)
    for _ in range(3):
        mujoco.mj_forward(env.model, env.data)
        sensor.update(float(env.data.time))
        env.step()
    sample = sensor.sample(float(env.data.time))
    assert sample.time_s == pytest.approx(start)
    assert sample.wrench_local.shape == (6,)
    assert np.isfinite(sample.wrench_local).all()
