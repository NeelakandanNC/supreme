"""On-disk rollout store.

Layout (one directory per dataset)::

    runs/data/TMaze-v0_s0/
        obs.npy        (N, T, H, W, 3) uint8    memory-mapped
        action.npy     (N, T, A)       float32  memory-mapped
        meta.npz       reward, terminal, mask, length, phase, cue,
                       correct_action, memory_critical, delay_since_cue, variant
        manifest.json  env id/kwargs, policy, seeds, shapes, byte sizes

Why memory-mapped ``.npy`` instead of sharded ``.npz``
------------------------------------------------------
The frame tensor is the only large object in this study (about 1.5 GB at the
default 400 x 300 x 64 x 64 x 3).  Memory-mapping hands residency management to
the OS page cache, which is exactly right on a 16 GB machine: the VAE's random
frame access touches pages uniformly and the kernel evicts them without the
process ever holding 1.5 GB resident.  A compressed shard scheme would instead
force us to decompress whole shards into the heap, which is what pushes a
laptop into swap.

Preallocating the file also lets several collector processes write disjoint row
ranges concurrently with no locking and no merge step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Per-step integer/boolean side-channels emitted by the environments.  These
#: are ground-truth labels for the benchmark probes; the model never sees them.
META_INT_KEYS = ("phase", "cue", "correct_action", "delay_since_cue", "variant")
META_BOOL_KEYS = ("memory_critical",)


@dataclass
class StoreSpec:
    n_rollouts: int
    steps: int
    image_size: int
    action_dim: int

    @property
    def obs_bytes(self) -> int:
        return self.n_rollouts * self.steps * self.image_size * self.image_size * 3

    def describe(self) -> str:
        return (f"{self.n_rollouts} rollouts x {self.steps} steps "
                f"@ {self.image_size}px -> {self.obs_bytes / 1e9:.2f} GB on disk")


class RolloutStore:
    """Create / open a rollout dataset directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------ create --
    @classmethod
    def create(cls, root: str | Path, spec: StoreSpec, manifest: dict[str, Any]) -> "RolloutStore":
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(
            store.obs_path, mode="w+", dtype=np.uint8,
            shape=(spec.n_rollouts, spec.steps, spec.image_size, spec.image_size, 3),
        )
        np.lib.format.open_memmap(
            store.action_path, mode="w+", dtype=np.float32,
            shape=(spec.n_rollouts, spec.steps, spec.action_dim),
        )
        payload = {"spec": spec.__dict__, **manifest}
        store.manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return store

    # -------------------------------------------------------------- paths --
    @property
    def obs_path(self) -> Path:
        return self.root / "obs.npy"

    @property
    def action_path(self) -> Path:
        return self.root / "action.npy"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.npz"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def latents_path(self) -> Path:
        return self.root / "latents.npz"

    def exists(self) -> bool:
        return self.obs_path.exists() and self.meta_path.exists()

    # --------------------------------------------------------------- io ---
    def open_obs(self, mode: str = "r") -> np.memmap:
        return np.load(self.obs_path, mmap_mode=mode)  # type: ignore[return-value]

    def open_actions(self, mode: str = "r") -> np.memmap:
        return np.load(self.action_path, mmap_mode=mode)  # type: ignore[return-value]

    def load_meta(self) -> dict[str, np.ndarray]:
        with np.load(self.meta_path) as data:
            return {k: data[k] for k in data.files}

    def save_meta(self, meta: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.meta_path, **meta)

    @property
    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @property
    def spec(self) -> StoreSpec:
        return StoreSpec(**self.manifest["spec"])

    # --------------------------------------------------------- splitting --
    def split_indices(self, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic train/val split *at the rollout level*.

        Splitting by rollout rather than by frame is essential: frames within an
        episode are near-duplicates, so a frame-level split leaks the validation
        set into training and makes both models look better than they are.
        """
        n = self.spec.n_rollouts
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(val_fraction * n)))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])
