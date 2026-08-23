"""Environment registry.

``make_env(cfg)`` is the single construction point used by data collection, by
the CMA-ES workers and by the benchmark, so every consumer sees byte-identical
observations.
"""
from __future__ import annotations

from typing import Any, Callable

from wmcore.envs.base import Env
from wmcore.envs.memory_envs import (
    PHASE_CUE,
    PHASE_DECISION,
    PHASE_DELAY,
    PHASE_FEEDBACK,
    SequenceRecallEnv,
    TMazeMemoryEnv,
)
from wmcore.envs.spaces import Box, Discrete
from wmcore.envs.wrappers import ActionRepeat, ResizeObservation, TimeLimit

_REGISTRY: dict[str, Callable[..., Env]] = {}


def register(env_id: str, builder: Callable[..., Env]) -> None:
    _REGISTRY[env_id] = builder


def registered_envs() -> list[str]:
    return sorted(_REGISTRY)


def _build_tmaze(**kwargs) -> Env:
    return TMazeMemoryEnv(**kwargs)


def _build_recall(**kwargs) -> Env:
    return SequenceRecallEnv(**kwargs)


def _build_car_racing(**kwargs) -> Env:
    from wmcore.envs.car_racing import make_car_racing

    return make_car_racing(**kwargs)


register("TMaze-v0", _build_tmaze)
register("SequenceRecall-v0", _build_recall)
register("CarRacing-v0", _build_car_racing)


def make_env(
    env_id: str,
    *,
    image_size: int = 64,
    max_episode_steps: int | None = None,
    action_repeat: int = 1,
    **kwargs: Any,
) -> Env:
    """Instantiate an environment by registry id.

    The synthetic environments render natively at ``image_size`` and manage
    their own episode length, so the resize/time-limit wrappers are only
    applied where they are actually needed.  That keeps the fast path fast.
    """
    if env_id not in _REGISTRY:
        raise KeyError(f"unknown env {env_id!r}; registered: {registered_envs()}")

    if env_id == "CarRacing-v0":
        return _REGISTRY[env_id](
            image_size=image_size,
            action_repeat=action_repeat,
            max_episode_steps=max_episode_steps or 300,
            **kwargs,
        )

    env = _REGISTRY[env_id](image_size=image_size, **kwargs)
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)
    return env


def make_env_from_config(cfg) -> Env:
    """Build from an :class:`~wmcore.config.EnvConfig`."""
    return make_env(
        cfg.id,
        image_size=cfg.image_size,
        max_episode_steps=cfg.max_episode_steps,
        action_repeat=cfg.action_repeat,
        **cfg.kwargs,
    )


__all__ = [
    "Env", "Box", "Discrete",
    "TMazeMemoryEnv", "SequenceRecallEnv",
    "PHASE_CUE", "PHASE_DELAY", "PHASE_DECISION", "PHASE_FEEDBACK",
    "ActionRepeat", "ResizeObservation", "TimeLimit",
    "make_env", "make_env_from_config", "register", "registered_envs",
]
