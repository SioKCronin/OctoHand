"""Discrete action wrapper so the original DQN agent has something to drive."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteMotionWrapper(gym.Wrapper):
    """Integrate small nudges into the continuous OctoHand command.

    Action 0 holds the current command. The rest come in pairs:
    ±x, ±y, ±z, ±yaw, then ±curl for each tentacle.
    """

    def __init__(self, env: gym.Env, step: float = 0.25):
        super().__init__(env)
        self.step_size = float(step)
        n_curl = env.unwrapped.n_tentacles
        self._nudges: list[tuple[int, float] | None] = [None]
        for index in range(4 + n_curl):
            self._nudges.append((index, self.step_size))
            self._nudges.append((index, -self.step_size))
        self.action_space = spaces.Discrete(len(self._nudges))
        self.command = np.zeros(env.unwrapped.n_actions, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.command = np.asarray(self.env.unwrapped.home_action(), dtype=np.float64)
        return obs, info

    def step(self, action):
        action = int(action)
        if action < 0 or action >= len(self._nudges):
            raise ValueError(f"Discrete action {action} is outside 0..{len(self._nudges) - 1}.")
        nudge = self._nudges[action]
        if nudge is not None:
            index, delta = nudge
            self.command[index] = float(np.clip(self.command[index] + delta, -1.0, 1.0))
        return self.env.step(self.command)
