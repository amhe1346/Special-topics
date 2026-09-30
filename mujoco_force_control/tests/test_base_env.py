from envs.base_env import BaseEnv


def test_panda_model_loads_and_steps():
    env = BaseEnv()
    env.step(10)

    expected_time = 10 * env.model.opt.timestep
    assert abs(env.data.time - expected_time) < 1e-12


def test_reading_missing_sensor_has_clear_error():
    env = BaseEnv()

    try:
        env.read_sensor("not_a_sensor")
    except KeyError as error:
        assert "not_a_sensor" in str(error)
    else:
        raise AssertionError("reading an undefined sensor should raise KeyError")