"""Linear policy used by particle-swarm search."""

from __future__ import annotations

import numpy as np


def n_params(n_obs: int, n_act: int) -> int:
    return n_obs * n_act + n_act


class LinearPolicy:
    """``tanh(W o + b)``. Parameters are stored row-major as ``[W | b]``."""

    def __init__(self, params: np.ndarray, n_obs: int, n_act: int):
        params = np.asarray(params, dtype=np.float64).ravel()
        expected = n_params(n_obs, n_act)
        if params.size != expected:
            raise ValueError(f"Expected {expected} parameters, got {params.size}.")
        self.W = params[: n_obs * n_act].reshape(n_act, n_obs)
        self.b = params[n_obs * n_act :]

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).ravel()
        return np.tanh(self.W @ obs + self.b)
