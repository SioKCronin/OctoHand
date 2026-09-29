"""Deep Q-network in NumPy.

The original agent was a Keras MLP with a replay buffer and epsilon-greedy
actions, but it bootstrapped from the same network it was updating. This
version keeps that architecture (two 24-unit ReLU layers, Adam, gamma 0.95)
and adds a target network.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class _MLP:
    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator):
        self.sizes = sizes
        self.W = []
        self.b = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def copy_from(self, other: _MLP) -> None:
        for W, b, src_w, src_b in zip(self.W, self.b, other.W, other.b):
            W[...] = src_w
            b[...] = src_b

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0.0)
            hs.append(z)
        return hs

    def parameters(self) -> list[np.ndarray]:
        params = []
        for W, b in zip(self.W, self.b):
            params.extend((W, b))
        return params


class _Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (grad * grad)
            mhat = self.m[i] / (1.0 - self.b1**self.t)
            vhat = self.v[i] / (1.0 - self.b2**self.t)
            param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def _backward(model: _MLP, hs: list[np.ndarray], dloss: np.ndarray):
    grads_w = [None] * len(model.W)
    grads_b = [None] * len(model.b)
    delta = dloss
    for i in reversed(range(len(model.W))):
        grads_w[i] = hs[i].T @ delta
        grads_b[i] = delta.sum(axis=0)
        if i == 0:
            break
        delta = delta @ model.W[i].T
        delta = delta * (hs[i] > 0.0)
    grads = []
    for W, b in zip(grads_w, grads_b):
        grads.extend((W, b))
    return grads


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        memory_size: int = 2000,
        target_interval: int = 200,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.memory = deque(maxlen=memory_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.learning_rate = float(learning_rate)
        self.target_interval = int(target_interval)
        self.rng = np.random.default_rng(seed)
        sizes = (self.state_size, 24, 24, self.action_size)
        self.model = _MLP(sizes, self.rng)
        self.target = _MLP(sizes, self.rng)
        self.target.copy_from(self.model)
        self._adam = _Adam(self.model.parameters(), lr=self.learning_rate)
        self._updates = 0

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append(
            (
                np.asarray(state, dtype=np.float64).ravel(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float64).ravel(),
                float(done),
            )
        )

    def act(self, state) -> int:
        if self.rng.random() <= self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        x = np.asarray(state, dtype=np.float64).reshape(1, -1)
        q = self.model.forward(x)[-1]
        return int(np.argmax(q[0]))

    def replay(self, batch_size: int = 32) -> float | None:
        if len(self.memory) < batch_size:
            return None
        idxs = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in idxs]
        states = np.stack([row[0] for row in batch])
        actions = np.array([row[1] for row in batch], dtype=np.int64)
        rewards = np.array([row[2] for row in batch])
        next_states = np.stack([row[3] for row in batch])
        dones = np.array([row[4] for row in batch])

        hs = self.model.forward(states)
        q = hs[-1]
        next_q = self.target.forward(next_states)[-1].max(axis=1)
        target = rewards + (1.0 - dones) * self.gamma * next_q
        taken = q[np.arange(batch_size), actions]
        diff = taken - target
        dloss = np.zeros_like(q)
        dloss[np.arange(batch_size), actions] = (2.0 / batch_size) * diff
        grads = _backward(self.model, hs, dloss)
        self._clip_grads(grads, max_norm=5.0)
        self._adam.step(self.model.parameters(), grads)
        self._updates += 1
        if self._updates % self.target_interval == 0:
            self.target.copy_from(self.model)
        return float(np.mean(diff**2))

    def decay(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def _clip_grads(grads: list[np.ndarray], max_norm: float) -> None:
        total = 0.0
        for grad in grads:
            total += float(np.sum(grad * grad))
        norm = np.sqrt(total)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-8)
            for grad in grads:
                grad *= scale
