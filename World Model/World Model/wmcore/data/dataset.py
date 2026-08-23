"""Torch datasets over a :class:`~wmcore.data.store.RolloutStore`.

Two consumers, two datasets:

``FrameDataset``
    i.i.d. 64x64 frames for training V.  Reads straight from the memmap, so
    resident memory stays flat no matter how large the store is.
``LatentSequenceDataset``
    fixed-length windows of *latent* vectors for training M.  Because latents
    are precomputed once (Ha & Schmidhuber do the same), the entire sequence
    training set is ~30 MB and lives comfortably in RAM -- which is why the
    memory-layer ablation is cheap to run many times and many seeds, and why
    this study is feasible on a laptop at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from wmcore.data.store import RolloutStore


# ---------------------------------------------------------------- frames --
class FrameDataset(Dataset):
    """Individual frames as float32 CHW in [0, 1]."""

    def __init__(self, store: RolloutStore, rollout_indices: np.ndarray,
                 max_frames: int | None = None, seed: int = 0):
        self.store = store
        self.obs = store.open_obs("r")
        meta = store.load_meta()
        mask = meta["mask"][rollout_indices]

        rows, cols = np.nonzero(mask)
        self.index = np.stack([rollout_indices[rows], cols], axis=1).astype(np.int32)

        if max_frames is not None and len(self.index) > max_frames:
            # Subsample *uniformly across the whole dataset* rather than
            # truncating, so the frame distribution is unchanged.
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(self.index), size=max_frames, replace=False)
            self.index = self.index[np.sort(keep)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        n, t = self.index[i]
        frame = np.asarray(self.obs[n, t], dtype=np.uint8)
        x = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
        return x


# ------------------------------------------------------------- sequences --
@dataclass(frozen=True)
class Segment:
    """A contiguous, single-episode span inside one stored rollout row."""

    row: int
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def episode_segments(store: RolloutStore, rollout_indices: np.ndarray) -> list[Segment]:
    """Split stored rows into single-episode segments.

    A row can contain more than one episode (CarRacing terminates early and the
    collector immediately restarts).  Training windows must never straddle a
    boundary: doing so would ask the memory model to carry state across a hard
    reset and would silently penalise whichever backbone forgets *less*.
    """
    meta = store.load_meta()
    terminal, mask = meta["terminal"], meta["mask"]
    segments: list[Segment] = []
    for row in rollout_indices:
        row = int(row)
        valid = np.nonzero(mask[row])[0]
        if valid.size == 0:
            continue
        start = int(valid[0])
        ends = np.nonzero(terminal[row])[0]
        boundaries = list(ends + 1) + [int(valid[-1]) + 1]
        for b in sorted(set(int(x) for x in boundaries)):
            if b > start:
                segments.append(Segment(row, start, b))
                start = b
    return segments


class LatentSequenceDataset(Dataset):
    """Windows of ``(z_t, a_t) -> z_{t+1}`` for training the memory model.

    Returns a dict of tensors:

    ``z``      (L, latent)   sampled latent, z ~ N(mu, sigma) -- resampled every
                             epoch, which is the regularisation Ha & Schmidhuber
                             rely on to stop M from overfitting a point estimate
    ``z_next`` (L, latent)   target latents (same sampling)
    ``action`` (L, A)
    ``reward`` (L,)
    ``done``   (L,)
    ``memory_critical`` (L,) bool -- steps whose target is only predictable from
                             memory; the benchmark reports NLL restricted to these
    ``cue``    (L,)          ground-truth latent variable, for probes only
    ``delay``  (L,)          steps since the cue was shown, for probes only
    ``correct_action`` (L,)  the action the environment would reward, for the
                             decision-accuracy metric only
    """

    def __init__(
        self,
        store: RolloutStore,
        rollout_indices: np.ndarray,
        seq_len: int,
        *,
        sample_latents: bool = True,
        seed: int = 0,
        stride: int | None = None,
    ):
        self.store = store
        self.seq_len = int(seq_len)
        self.sample_latents = sample_latents
        self.rng = np.random.default_rng(seed)

        with np.load(store.latents_path) as data:
            self.mu = data["mu"]           # (N, T, D) float32
            self.logvar = data["logvar"]   # (N, T, D) float32

        meta = store.load_meta()
        self.action = store.open_actions("r")
        self.reward = meta["reward"]
        self.terminal = meta["terminal"]
        self.memory_critical = meta["memory_critical"]
        self.cue = meta["cue"]
        self.delay = meta["delay_since_cue"]
        self.correct_action = meta["correct_action"]

        stride = stride or max(1, self.seq_len // 2)
        # (row, start, segment_end) -- the segment end is carried so that a
        # window can never read past the episode it belongs to.
        self.windows: list[tuple[int, int, int]] = []
        for seg in episode_segments(store, rollout_indices):
            # +1 because the last element of a window needs a successor target.
            last_start = seg.end - self.seq_len - 1
            if last_start < seg.start:
                if seg.length > 2:  # short episode: keep one truncated window
                    self.windows.append((seg.row, seg.start, seg.end))
                continue
            starts = list(range(seg.start, last_start + 1, stride))
            # Always include the tail window.  Without this, a stride that does
            # not divide the episode length silently drops the final steps --
            # and in the memory environments the final steps are precisely the
            # decision and feedback transitions the whole study is about.
            if starts[-1] != last_start:
                starts.append(last_start)
            self.windows.extend((seg.row, s, seg.end) for s in starts)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def latent_dim(self) -> int:
        return int(self.mu.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[-1])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row, start, seg_end = self.windows[i]
        L = self.seq_len
        # Clamp to the episode, not just to the array: a truncated window at the
        # tail of a short episode would otherwise read the first frames of the
        # NEXT episode stored in the same row and present them as a continuous
        # trajectory.  That silently rewards whichever backbone forgets faster.
        end = min(start + L, seg_end - 1, self.mu.shape[1] - 1)
        L_eff = end - start
        if L_eff <= 0:
            raise IndexError(f"degenerate window row={row} start={start} end={end}")

        mu = self.mu[row, start:end + 1]
        logvar = self.logvar[row, start:end + 1]
        if self.sample_latents:
            eps = self.rng.standard_normal(mu.shape).astype(np.float32)
            z_full = mu + np.exp(0.5 * logvar) * eps
        else:
            z_full = mu

        out = {
            "z": _pad(z_full[:-1], L),
            "z_next": _pad(z_full[1:], L),
            "action": _pad(np.asarray(self.action[row, start:end]), L),
            "reward": _pad(self.reward[row, start:end], L),
            "done": _pad(self.terminal[row, start:end].astype(np.float32), L),
            "mask": _pad(np.ones(L_eff, dtype=np.float32), L),
            "memory_critical": _pad(self.memory_critical[row, start:end].astype(np.float32), L),
            "cue": _pad(self.cue[row, start:end].astype(np.int64), L, fill=-1),
            "delay": _pad(self.delay[row, start:end].astype(np.int64), L, fill=-1),
            "correct_action": _pad(self.correct_action[row, start:end].astype(np.int64),
                                   L, fill=-1),
        }
        # np.array(..., copy=True): several of these are slices of a read-only
        # memmap, and torch.from_numpy on a non-writable buffer warns and yields
        # a tensor whose in-place ops are undefined behaviour.
        return {k: torch.from_numpy(np.array(v, copy=True)) for k, v in out.items()}


def _pad(arr: np.ndarray, length: int, fill: float = 0.0) -> np.ndarray:
    """Right-pad along axis 0 to ``length`` (windows at an episode tail)."""
    if arr.shape[0] == length:
        return arr
    pad_shape = (length - arr.shape[0],) + arr.shape[1:]
    return np.concatenate([arr, np.full(pad_shape, fill, dtype=arr.dtype)], axis=0)
