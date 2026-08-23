"""Observation / timing wrappers shared by every environment.

Kept minimal and dependency-free so the same code path serves both the numpy
memory environments and the gymnasium-backed CarRacing.
"""
from __future__ import annotations

import numpy as np

from wmcore.envs.base import Env


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 frame to ``size x size``.

    Uses PIL bilinear when available (better downsampling of CarRacing's 96x96
    track texture), otherwise a pure-numpy nearest-neighbour fallback so the
    package still runs on a bare interpreter.
    """
    if frame.shape[0] == size and frame.shape[1] == size:
        return np.ascontiguousarray(frame)
    try:
        from PIL import Image

        return np.asarray(
            Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8
        )
    except ImportError:
        ys = (np.linspace(0, frame.shape[0] - 1, size)).astype(np.int64)
        xs = (np.linspace(0, frame.shape[1] - 1, size)).astype(np.int64)
        return np.ascontiguousarray(frame[ys][:, xs])


class Wrapper(Env):
    """Pass-through base wrapper."""

    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_shape = env.observation_shape
        self.metadata = env.metadata

    def reset(self, *, seed: int | None = None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, item):
        # Forward unknown attributes (e.g. `episode_length`) to the inner env.
        return getattr(self.__dict__["env"], item)


class ResizeObservation(Wrapper):
    """Force every observation to ``size x size x 3`` uint8."""

    def __init__(self, env: Env, size: int = 64):
        super().__init__(env)
        self.size = int(size)
        self.observation_shape = (self.size, self.size, 3)

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return resize_frame(obs, self.size), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return resize_frame(obs, self.size), r, term, trunc, info


class ActionRepeat(Wrapper):
    """Hold each action for ``k`` frames, summing reward.

    On CarRacing this is the cheapest way to shorten the effective horizon:
    k=2 halves the number of frames the VAE and M model must chew through with
    almost no loss of control fidelity.
    """

    def __init__(self, env: Env, k: int = 1):
        super().__init__(env)
        self.k = max(1, int(k))

    def step(self, action):
        total = 0.0
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total += r
            if term or trunc:
                break
        return obs, total, term, trunc, info


class TimeLimit(Wrapper):
    """Truncate after ``max_steps`` interactions."""

    def __init__(self, env: Env, max_steps: int):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self._elapsed = 0

    def reset(self, *, seed: int | None = None):
        self._elapsed = 0
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.max_steps:
            trunc = True
        return obs, r, term, trunc, info
