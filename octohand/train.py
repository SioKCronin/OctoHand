"""Train a policy on OctoHand and log the return."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from octohand.agents.dqn import DQNAgent
from octohand.agents.policy import LinearPolicy
from octohand.agents.pso import optimize
from octohand.env import OctoHandEnv
from octohand.scores import log_score
from octohand.wrappers import DiscreteMotionWrapper


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an OctoHand policy.")
    parser.add_argument("algo", choices=("pso", "dqn"))
    parser.add_argument("--task", choices=("block", "egg", "pen"), default="block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", type=Path, default=Path("data/scores.db"))
    parser.add_argument("--out", type=Path, default=None, help="Where to save the best PSO weights.")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="DQN episodes (default 30), or PSO episodes per particle (default 1).",
    )
    args = parser.parse_args(argv)

    if args.algo == "pso":
        episodes = 1 if args.episodes is None else args.episodes
        weights, score, _history = optimize(
            lambda: OctoHandEnv(task=args.task),
            n_particles=args.particles,
            iterations=args.iters,
            episodes=episodes,
            seed=args.seed,
        )
        log_score(score, args.scores)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.out, weights)
            print(f"saved {args.out}")
        print(f"best return {score:.3f}")
        return

    episodes = 30 if args.episodes is None else args.episodes
    env = DiscreteMotionWrapper(OctoHandEnv(task=args.task))
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=args.seed)
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=args.seed + episode)
            done = False
            total = 0.0
            while not done:
                action = agent.act(obs)
                nxt, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                agent.remember(obs, action, reward, nxt, done)
                agent.replay()
                obs = nxt
                total += float(reward)
            agent.decay()
            log_score(total, args.scores)
            print(f"episode {episode + 1:03d}  return {total:7.2f}  epsilon {agent.epsilon:.3f}")
    finally:
        env.close()


def rollout(weights_path: Path, task: str = "block", seed: int = 0) -> float:
    """Evaluate a saved linear policy. Used by tests and ad-hoc checks."""
    env = OctoHandEnv(task=task)
    weights = np.load(weights_path)
    policy = LinearPolicy(weights, env.observation_space.shape[0], env.action_space.shape[0])
    obs, _info = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, _info = env.step(policy.act(obs))
        total += float(reward)
        done = bool(terminated or truncated)
    env.close()
    return total


if __name__ == "__main__":
    main()
