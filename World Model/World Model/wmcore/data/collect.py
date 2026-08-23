"""Rollout collection.

Runs ``n_rollouts`` episodes and writes them into a :class:`RolloutStore`.
Parallelised with ``spawn`` processes -- the macOS default -- each writing a
disjoint slice of the preallocated memmap, so there is no merge step and peak
RAM is one episode per worker rather than the whole dataset.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from wmcore.config import Config
from wmcore.data.policies import make_policy
from wmcore.data.store import META_BOOL_KEYS, META_INT_KEYS, RolloutStore, StoreSpec
from wmcore.envs import make_env_from_config
from wmcore.utils.logging_utils import get_logger
from wmcore.utils.parallel import parallel_map

log = get_logger(__name__)


def dataset_dir(cfg: Config) -> Path:
    """Datasets are keyed by env + collection settings, never by model.

    Both models read the *same* directory.  Regenerating data per model would
    be the easiest way to accidentally destroy the comparison.
    """
    kw = "_".join(f"{k}{v}" for k, v in sorted(cfg.env.kwargs.items()))
    tag = f"{cfg.env.id}{'_' + kw if kw else ''}_n{cfg.data.n_rollouts}" \
          f"_t{cfg.data.steps_per_rollout}_p{cfg.data.policy}_s{cfg.seed}"
    return Path(cfg.data.root) / tag


def _episode_worker(args: tuple) -> tuple[int, int]:
    """Collect rollouts ``[lo, hi)`` into an existing store.  Runs in a subprocess."""
    root, cfg_dict, lo, hi, base_seed = args
    cfg = Config.from_dict(cfg_dict)
    store = RolloutStore(root)
    spec = store.spec

    obs_mm = store.open_obs("r+")
    act_mm = store.open_actions("r+")

    env = make_env_from_config(cfg.env)
    n_written = 0
    local_meta: dict[str, np.ndarray] = _empty_meta(hi - lo, spec.steps)

    for i in range(lo, hi):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        policy = make_policy(cfg.data.policy, env.action_space, rng, env_id=cfg.env.id)

        obs, info = env.reset(seed=seed)
        row = i - lo
        for t in range(spec.steps):
            obs_mm[i, t] = obs
            _record_info(local_meta, row, t, info)

            action = policy(obs, t)
            act_mm[i, t] = env.action_space.to_vector(action)

            obs, reward, terminated, truncated, info = env.step(action)
            local_meta["reward"][row, t] = reward
            local_meta["terminal"][row, t] = terminated
            local_meta["mask"][row, t] = True
            # Number of stored steps in this row.  A row can hold several
            # episodes (see below), so this is the fill level, not an episode
            # length -- episode boundaries live in `terminal`.
            local_meta["length"][row] = t + 1

            if terminated or truncated:
                if t + 1 < spec.steps:
                    # Episode ended early: start a fresh one so the tail of the
                    # row is real data rather than padding.  `terminal` marks the
                    # boundary, and the sequence sampler never crosses it.
                    obs, info = env.reset(seed=seed * 7919 + t)
        n_written += 1

    env.close()
    obs_mm.flush()
    act_mm.flush()
    np.savez(Path(root) / f"_meta_part_{lo}_{hi}.npz", **local_meta)
    return lo, n_written


def _empty_meta(n: int, t: int) -> dict[str, np.ndarray]:
    meta = {
        "reward": np.zeros((n, t), dtype=np.float32),
        "terminal": np.zeros((n, t), dtype=bool),
        "mask": np.zeros((n, t), dtype=bool),
        "length": np.zeros((n,), dtype=np.int32),
    }
    for key in META_INT_KEYS:
        meta[key] = np.full((n, t), -1, dtype=np.int16)
    for key in META_BOOL_KEYS:
        meta[key] = np.zeros((n, t), dtype=bool)
    return meta


def _record_info(meta: dict[str, np.ndarray], row: int, t: int, info: dict[str, Any]) -> None:
    for key in META_INT_KEYS:
        if key in info:
            meta[key][row, t] = int(info[key])
    for key in META_BOOL_KEYS:
        if key in info:
            meta[key][row, t] = bool(info[key])


def collect(cfg: Config, *, force: bool = False, n_workers: int | None = None) -> RolloutStore:
    """Collect (or reuse) the dataset described by ``cfg``."""
    root = dataset_dir(cfg)
    store = RolloutStore(root)
    if store.exists() and not force:
        log.info("reusing dataset at %s (%s)", root, store.spec.describe())
        return store

    probe_env = make_env_from_config(cfg.env)
    action_dim = probe_env.action_dim
    steps = cfg.data.steps_per_rollout
    # Synthetic memory envs know their own length; never collect padding.
    if hasattr(probe_env, "episode_length"):
        steps = min(steps, int(probe_env.episode_length))
    probe_env.close()

    spec = StoreSpec(cfg.data.n_rollouts, steps, cfg.env.image_size, action_dim)
    log.info("collecting %s", spec.describe())

    store = RolloutStore.create(root, spec, manifest={
        "env": {"id": cfg.env.id, "kwargs": cfg.env.kwargs,
                "image_size": cfg.env.image_size,
                "action_repeat": cfg.env.action_repeat},
        "policy": cfg.data.policy,
        "seed": cfg.seed,
        "n_rollouts": cfg.data.n_rollouts,
    })

    n_workers = n_workers or max(1, min(cfg.controller.n_workers, os.cpu_count() or 1))
    bounds = np.linspace(0, cfg.data.n_rollouts, n_workers + 1).astype(int)
    jobs = [(str(root), cfg.to_dict(), int(bounds[i]), int(bounds[i + 1]), cfg.seed * 100_003)
            for i in range(n_workers) if bounds[i + 1] > bounds[i]]

    results = parallel_map(_episode_worker, jobs, n_workers=len(jobs),
                           label="rollout collection")
    log.info("collected %d rollouts across %d workers", sum(r[1] for r in results), len(jobs))

    _merge_meta(store, spec)
    return store


def _merge_meta(store: RolloutStore, spec: StoreSpec) -> None:
    """Stitch the per-worker meta parts into one meta.npz and delete the parts."""
    parts = sorted(store.root.glob("_meta_part_*.npz"),
                   key=lambda p: int(p.stem.split("_")[-2]))
    merged = _empty_meta(spec.n_rollouts, spec.steps)
    for part in parts:
        lo = int(part.stem.split("_")[-2])
        with np.load(part) as data:
            n = data["mask"].shape[0]
            for key in merged:
                merged[key][lo:lo + n] = data[key]
    store.save_meta(merged)
    for part in parts:
        part.unlink()
    log.info("wrote %s", store.meta_path)


def summarise(store: RolloutStore) -> dict:
    """Quick sanity report -- printed after collection and stored in the manifest."""
    meta = store.load_meta()
    spec = store.spec
    ret = meta["reward"].sum(axis=1)
    out = {
        "n_rollouts": spec.n_rollouts,
        "steps": spec.steps,
        "frames": int(meta["mask"].sum()),
        "disk_gb": round(spec.obs_bytes / 1e9, 3),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "memory_critical_steps": int(meta["memory_critical"].sum()),
    }
    if (meta["cue"] >= 0).any():
        cues = meta["cue"][meta["cue"] >= 0]
        out["cue_entropy_bits"] = float(_entropy_bits(cues))
    return out


def _entropy_bits(values: np.ndarray) -> float:
    counts = np.bincount(values.astype(np.int64))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())
