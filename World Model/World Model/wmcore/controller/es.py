"""Evolution strategies for training C.

Two implementations behind one interface:

``CMAES``       thin wrapper over the ``cma`` package (Hansen's reference
                implementation), which is what Ha & Schmidhuber use.
``OpenAIES``    a ~40-line isotropic-Gaussian ES with rank-based fitness
                shaping and mirrored sampling, used as a fallback so the repo
                runs with zero optional dependencies.

Both are seeded and both log the same fields, so a run is reproducible and the
two legs of the study can share a search-budget table.
"""
from __future__ import annotations

import abc

import numpy as np


class Strategy(abc.ABC):
    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Return ``(population, n_params)`` candidate solutions."""

    @abc.abstractmethod
    def tell(self, solutions: np.ndarray, fitness: np.ndarray) -> None:
        """Report fitness (higher is better)."""

    @abc.abstractmethod
    def best(self) -> np.ndarray:
        """Current recommendation."""


class CMAES(Strategy):
    """Covariance Matrix Adaptation ES.

    CMA-ES scales as O(n^2) in the number of parameters, which is fine here:
    C has under a thousand parameters even with a 256-d memory feature.
    """

    def __init__(self, n_params: int, population: int = 32, sigma: float = 0.5,
                 seed: int = 0, x0: np.ndarray | None = None):
        try:
            import cma
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install cma  (or set controller.algo='openai_es')"
            ) from exc
        self._es = cma.CMAEvolutionStrategy(
            (x0 if x0 is not None else np.zeros(n_params)).tolist(),
            sigma,
            {"popsize": population, "seed": seed + 1, "verbose": -9},
        )
        self._last: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._last = np.asarray(self._es.ask(), dtype=np.float64)
        return self._last

    def tell(self, solutions: np.ndarray, fitness: np.ndarray) -> None:
        # `cma` minimises.
        self._es.tell(list(solutions), list(-np.asarray(fitness, dtype=np.float64)))

    def best(self) -> np.ndarray:
        xfav = self._es.result.xfavorite
        return np.asarray(xfav if xfav is not None else self._es.result.xbest,
                          dtype=np.float64)


class OpenAIES(Strategy):
    """Isotropic-Gaussian ES with mirrored sampling and rank shaping.

    Dependency-free fallback.  Mirrored sampling (each perturbation used with
    both signs) roughly halves gradient-estimate variance, which matters a lot
    when each fitness evaluation costs several environment episodes.
    """

    def __init__(self, n_params: int, population: int = 32, sigma: float = 0.1,
                 lr: float = 0.03, seed: int = 0, x0: np.ndarray | None = None,
                 weight_decay: float = 0.005):
        if population % 2:
            population += 1
        self.n_params = n_params
        self.population = population
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        self.mu = np.zeros(n_params) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        self._eps: np.ndarray | None = None
        # Adam state -- plain SGD on an ES gradient estimate is noticeably worse.
        self._m = np.zeros(n_params)
        self._v = np.zeros(n_params)
        self._t = 0

    def ask(self) -> np.ndarray:
        half = self.population // 2
        eps = self.rng.standard_normal((half, self.n_params))
        self._eps = np.concatenate([eps, -eps], axis=0)
        return self.mu[None, :] + self.sigma * self._eps

    def tell(self, solutions: np.ndarray, fitness: np.ndarray) -> None:
        assert self._eps is not None, "call ask() before tell()"
        ranks = _centered_ranks(np.asarray(fitness, dtype=np.float64))
        grad = (self._eps.T @ ranks) / (self.population * self.sigma)
        grad -= self.weight_decay * self.mu

        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * grad * grad
        m_hat = self._m / (1 - b1 ** self._t)
        v_hat = self._v / (1 - b2 ** self._t)
        self.mu += self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def best(self) -> np.ndarray:
        return self.mu.copy()


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """Map fitness to ranks in [-0.5, 0.5]; makes ES invariant to reward scale."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x), dtype=np.float64)
    return ranks / max(1, len(x) - 1) - 0.5


def make_strategy(algo: str, n_params: int, *, population: int, sigma: float,
                  seed: int, x0: np.ndarray | None = None) -> Strategy:
    if algo == "cma":
        try:
            return CMAES(n_params, population, sigma, seed, x0)
        except ImportError:
            return OpenAIES(n_params, population, sigma, seed=seed, x0=x0)
    if algo == "openai_es":
        return OpenAIES(n_params, population, sigma, seed=seed, x0=x0)
    raise ValueError(f"unknown ES algorithm {algo!r}")
