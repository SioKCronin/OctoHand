"""Particle swarm search for a linear OctoHand policy.

The original ``agent/pso.py`` swarm-stepped a linear CartPole policy and never
terminated. This keeps that update — inertia, a pull toward each particle's
best, and a pull toward the swarm's best — and evaluates it on OctoHand.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from octohand.agents.policy import LinearPolicy, n_params


def _rollout(env, policy: LinearPolicy, seed: int) -> float:
    obs, _info = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, _info = env.step(policy.act(obs))
        total += float(reward)
        done = bool(terminated or truncated)
    return total


def optimize(
    env_fn: Callable,
    n_particles: int = 12,
    iterations: int = 15,
    episodes: int = 1,
    seed: int = 0,
    inertia: float = 0.72,
    cognitive: float = 1.45,
    social: float = 1.45,
    verbose: bool = True,
):
    """Maximize mean episode return. Returns ``(best_params, best_return, history)``."""
    env = env_fn()
    n_obs = int(env.observation_space.shape[0])
    n_act = int(env.action_space.shape[0])
    dim = n_params(n_obs, n_act)
    rng = np.random.default_rng(seed)

    pos = rng.normal(0.0, 0.12, size=(n_particles, dim))
    # Bias curl outputs toward open so the swarm does not start clamped shut.
    bias = n_obs * n_act
    pos[:, bias + 4 :] = -0.7
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_val = np.full(n_particles, -np.inf)
    gbest = pos[0].copy()
    gbest_val = -np.inf
    history: list[float] = []

    try:
        for it in range(iterations):
            seeds = [seed + 1000 + it * episodes + k for k in range(episodes)]
            values = np.empty(n_particles, dtype=np.float64)
            for i in range(n_particles):
                policy = LinearPolicy(pos[i], n_obs, n_act)
                returns = [_rollout(env, policy, s) for s in seeds]
                values[i] = float(np.mean(returns))
                if values[i] > pbest_val[i]:
                    pbest_val[i] = values[i]
                    pbest[i] = pos[i].copy()
                if values[i] > gbest_val:
                    gbest_val = float(values[i])
                    gbest = pos[i].copy()
            history.append(float(values.mean()))
            if verbose:
                print(
                    f"iter {it + 1:02d}/{iterations}  "
                    f"mean {values.mean():7.3f}  best {gbest_val:7.3f}"
                )
            r1 = rng.random(pos.shape)
            r2 = rng.random(pos.shape)
            vel = inertia * vel + cognitive * r1 * (pbest - pos) + social * r2 * (gbest - pos)
            vel = np.clip(vel, -0.35, 0.35)
            pos = np.clip(pos + vel, -2.5, 2.5)
    finally:
        env.close()
    return gbest, gbest_val, history
