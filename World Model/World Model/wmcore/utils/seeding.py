"""Deterministic seeding helpers.

Reproducibility matters more than usual here: the headline claim of the study
is a *difference* between two models, so run-to-run variance has to be
controlled and reported.  Every entry point takes a ``--seed`` and every
reported number should be aggregated over >= 3 seeds.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (CPU + MPS/CUDA).

    Parameters
    ----------
    seed:
        Base seed.
    deterministic:
        If True, ask torch for deterministic kernels where available.  On MPS
        this is mostly a no-op but it costs nothing and helps on CPU.

    Returns
    -------
    The seed, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # allow non-torch utilities to be used standalone
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # `warn_only` keeps us running when a kernel has no deterministic
        # implementation (several MPS kernels do not).
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for torch DataLoader so workers do not share a stream."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
