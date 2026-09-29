import pytest

from octohand.env import OctoHandEnv
from octohand.scripted import ScriptedPolicy


def _solved(task, seed):
    env = OctoHandEnv(task=task, settle_steps=8)
    policy = ScriptedPolicy()
    _obs, info = env.reset(seed=seed)
    for _ in range(env.max_steps):
        _obs, _reward, terminated, truncated, info = env.step(policy.act(env))
        if terminated or truncated:
            break
    env.close()
    return info


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scripted_block(seed):
    info = _solved("block", seed)
    assert info["success"], info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_egg(seed):
    info = _solved("egg", seed)
    assert info["success"], info


@pytest.mark.parametrize("seed", [0, 3, 5])
def test_scripted_pen(seed):
    info = _solved("pen", seed)
    assert info["success"], info
    assert info["align"] >= 0.8
