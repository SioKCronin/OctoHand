"""OctoHand: cephalopod-inspired soft-gripper tasks for reinforcement learning."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from octohand.env import OctoHandEnv
from octohand.scripted import ScriptedPolicy

__version__ = "0.1.0"

_ENVS = {
    "OctoHandBlock-v0": "block",
    "OctoHandEgg-v0": "egg",
    "OctoHandPen-v0": "pen",
}


def register_envs() -> None:
    existing = set(registry.keys())
    for env_id, task in _ENVS.items():
        if env_id in existing:
            continue
        register(
            id=env_id,
            entry_point="octohand.env:OctoHandEnv",
            kwargs={"task": task},
        )


def make(task: str = "block", **kwargs) -> OctoHandEnv:
    """Build an environment without going through Gymnasium's registry."""
    return OctoHandEnv(task=task, **kwargs)


register_envs()

__all__ = [
    "OctoHandEnv",
    "ScriptedPolicy",
    "make",
    "register_envs",
    "__version__",
]
