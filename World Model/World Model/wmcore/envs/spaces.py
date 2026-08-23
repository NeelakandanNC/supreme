"""Minimal action/observation spaces.

Deliberately not gymnasium's: the memory environments in this study are pure
numpy so that they run identically on any machine, install in zero seconds and
never break when gymnasium changes its API.  ``wmcore.envs.car_racing`` adapts
gymnasium spaces into these at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Discrete:
    n: int

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def vector_dim(self) -> int:
        """Width of the one-hot encoding fed to the memory model."""
        return self.n

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n))

    def to_vector(self, action) -> np.ndarray:
        vec = np.zeros(self.n, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def from_vector(self, vec: np.ndarray):
        return int(np.argmax(vec))


@dataclass(frozen=True)
class Box:
    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.low.shape)

    @property
    def vector_dim(self) -> int:
        return int(np.prod(self.low.shape))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high).astype(np.float32)

    def to_vector(self, action) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def from_vector(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vec, dtype=np.float32).reshape(self.shape),
                       self.low, self.high)
