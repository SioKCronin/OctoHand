import numpy as np
import gymnasium as gym
import pytest

import octohand
from octohand.env import OctoHandEnv
from octohand.model import build_model


@pytest.mark.parametrize("task", ["block", "egg", "pen"])
def test_reset_and_step(task):
    env = OctoHandEnv(task=task, settle_steps=0, max_steps=5)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert len(env.observation_labels()) == obs.shape[0]
    obs, reward, terminated, truncated, info = env.step(env.home_action())
    assert np.isfinite(reward)
    assert info["task"] == task
    env.close()


def test_same_seed_is_deterministic():
    def rollout():
        env = OctoHandEnv("block", settle_steps=0, max_steps=3, frame_skip=2)
        obs, _ = env.reset(seed=4)
        seen = [obs.copy()]
        for _ in range(3):
            obs, *_ = env.step(np.zeros(env.n_actions))
            seen.append(obs.copy())
        env.close()
        return seen

    first, second = rollout(), rollout()
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_registered_ids():
    env = gym.make("OctoHandEgg-v0")
    assert env.unwrapped.task == "egg"
    env.close()


def test_tentacle_count_changes_the_action_space():
    env = OctoHandEnv("block", n_tentacles=5, settle_steps=0)
    assert env.action_space.shape == (9,)
    env.reset(seed=0)
    env.close()


def test_model_accepts_the_documented_tentacle_range():
    build_model("block", 2)
    build_model("pen", 8)
    with pytest.raises(ValueError):
        build_model("block", 1)
    with pytest.raises(ValueError):
        build_model("cup", 3)


def test_pen_starts_on_its_side():
    env = OctoHandEnv("pen", settle_steps=0)
    env.reset(seed=1)
    assert abs(env.object_axis[2]) < 0.2
    assert env.object_position[2] < 0.05
    env.close()
