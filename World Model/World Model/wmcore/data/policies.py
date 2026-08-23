"""Exploration policies for rollout collection.

Ha & Schmidhuber collect CarRacing data with a uniformly random policy.  That
works there because the track scrolls past regardless of what you do, but it is
a poor default in general: on a continuous action space, i.i.d. uniform actions
average out to "do nothing" and the car never leaves the start tile.  We
therefore default to a temporally correlated (sticky / Ornstein-Uhlenbeck)
random policy, which covers far more of the state space per frame collected --
and frames collected is the binding constraint on a laptop.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from wmcore.envs.spaces import Box, Discrete

Policy = Callable[[np.ndarray, int], "np.ndarray | int"]


def uniform_random(action_space, rng: np.random.Generator) -> Policy:
    def policy(obs, t):
        return action_space.sample(rng)

    return policy


def sticky_random(action_space, rng: np.random.Generator, repeat_prob: float = 0.8) -> Policy:
    """Discrete: hold the previous action with probability ``repeat_prob``."""
    state = {"a": None}

    def policy(obs, t):
        if state["a"] is None or rng.random() > repeat_prob:
            state["a"] = action_space.sample(rng)
        return state["a"]

    return policy


def ornstein_uhlenbeck(action_space: Box, rng: np.random.Generator,
                       theta: float = 0.15, sigma: float = 0.35) -> Policy:
    """Continuous: correlated Brownian exploration clipped to the action box."""
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    mid = (low + high) / 2.0
    scale = (high - low) / 2.0
    state = {"x": np.zeros_like(mid)}

    def policy(obs, t):
        x = state["x"]
        x = x + theta * (0.0 - x) + sigma * rng.standard_normal(x.shape).astype(np.float32)
        state["x"] = np.clip(x, -2.0, 2.0)
        return np.clip(mid + scale * np.tanh(state["x"]), low, high).astype(np.float32)

    return policy


def car_racing_explore(action_space: Box, rng: np.random.Generator) -> Policy:
    """CarRacing-specific prior: mostly accelerate, occasionally steer/brake.

    A pure OU policy spends most of its time braking, which produces thousands
    of near-identical frames of a stationary car -- expensive and uninformative.
    Biasing the throttle keeps the car moving so the collected frames actually
    span the track distribution.
    """
    ou = ornstein_uhlenbeck(action_space, rng, theta=0.2, sigma=0.5)

    def policy(obs, t):
        a = np.asarray(ou(obs, t), dtype=np.float32).copy()
        a[1] = float(np.clip(rng.beta(3.0, 1.5), 0.0, 1.0))   # gas, skewed high
        a[2] = float(rng.random() < 0.05) * float(rng.random() * 0.6)  # rare brake
        return a

    return policy


def make_policy(name: str, action_space, rng: np.random.Generator, env_id: str = "") -> Policy:
    """Policy factory used by :mod:`wmcore.data.collect`."""
    if name == "random":
        if isinstance(action_space, Discrete):
            return uniform_random(action_space, rng)
        return car_racing_explore(action_space, rng) if "CarRacing" in env_id \
            else ornstein_uhlenbeck(action_space, rng)
    if name == "uniform":
        return uniform_random(action_space, rng)
    if name == "sticky":
        if isinstance(action_space, Discrete):
            return sticky_random(action_space, rng)
        return ornstein_uhlenbeck(action_space, rng)
    raise ValueError(f"unknown policy {name!r}")
