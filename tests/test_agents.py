import numpy as np

from octohand.agents.dqn import DQNAgent
from octohand.agents.policy import LinearPolicy, n_params
from octohand.agents.pso import optimize
from octohand.env import OctoHandEnv
from octohand.scores import log_score
from octohand.wrappers import DiscreteMotionWrapper


def test_linear_policy_shape():
    policy = LinearPolicy(np.zeros(n_params(4, 2)), 4, 2)
    action = policy.act(np.ones(4))
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)


def test_dqn_replay_updates_weights():
    agent = DQNAgent(4, 3, seed=0)
    before = agent.model.W[0].copy()
    rng = np.random.default_rng(1)
    for i in range(40):
        state = rng.normal(size=4)
        agent.remember(state, i % 3, float(state.sum()), rng.normal(size=4), False)
    loss = agent.replay(16)
    assert loss is not None and loss >= 0.0
    assert not np.allclose(before, agent.model.W[0])
    assert 0 <= agent.act(np.zeros(4)) < 3


def test_discrete_wrapper_and_pso_step():
    def factory():
        return OctoHandEnv("block", settle_steps=0, max_steps=8, frame_skip=2)

    wrapped = DiscreteMotionWrapper(factory())
    obs, _info = wrapped.reset(seed=0)
    obs, reward, terminated, truncated, _info = wrapped.step(1)
    assert np.isfinite(reward)
    wrapped.close()

    probe = factory()
    n_act = probe.action_space.shape[0]
    probe.close()
    best, value, history = optimize(factory, n_particles=2, iterations=1, episodes=1, seed=0, verbose=False)
    assert best.shape == (n_params(obs.shape[0], n_act),)
    assert np.isfinite(value)
    assert len(history) == 1


def test_score_log(tmp_path):
    path = tmp_path / "scores.db"
    log_score(1.5, path)
    log_score(-0.25, path)
    import sqlite3

    rows = sqlite3.connect(path).execute("SELECT scores FROM scores").fetchall()
    assert rows == [(1.5,), (-0.25,)]
