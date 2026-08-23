"""The environment protocol used everywhere in this repo.

A five-method gymnasium-compatible subset.  ``step`` returns the modern
five-tuple ``(obs, reward, terminated, truncated, info)``.

``info`` is where environments expose *ground-truth memory variables* -- the
cue colour, the query key, the correct answer.  Those labels never reach the
model; they are consumed only by the benchmark's linear probes, which ask
"is this fact still recoverable from the memory state after d steps?".  Having
them emitted by the environment is what makes the retention curves exact rather
than inferred.
"""
from __future__ import annotations

import abc
from typing import Any

import numpy as np

from wmcore.envs.spaces import Box, Discrete


class Env(abc.ABC):
    """Base class for every environment in the study."""

    action_space: Discrete | Box
    observation_shape: tuple[int, int, int] = (64, 64, 3)  # HWC uint8
    metadata: dict[str, Any] = {}

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Return ``(obs_uint8_hwc, info)``."""

    @abc.abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Return ``(obs, reward, terminated, truncated, info)``."""

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    # -- convenience ------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """Width of the action vector the memory model consumes."""
        return self.action_space.vector_dim

    def sample_action(self, rng: np.random.Generator):
        return self.action_space.sample(rng)
