"""Stage 2 -- freeze V and precompute latents for the whole dataset.

This is the step that makes the whole study affordable on a laptop.  After it
runs, training M touches a ~30 MB array of 32-d vectors instead of a 1.5 GB
frame store: an epoch drops from minutes to seconds, which is what lets us
afford many seeds, many sequence lengths and a continual-learning protocol.

We store ``mu`` and ``logvar`` rather than a single sampled z, so the sequence
dataset can resample ``z ~ N(mu, sigma)`` every epoch -- the regularisation Ha &
Schmidhuber rely on to keep M from overfitting a point estimate of V's output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wmcore.config import Config
from wmcore.data.collect import collect
from wmcore.data.store import RolloutStore
from wmcore.train.train_vae import load_vae, vae_checkpoint_path
from wmcore.utils import get_logger, pick_device, seed_everything

log = get_logger(__name__)


@torch.no_grad()
def encode_latents(cfg: Config, *, force: bool = False, batch_size: int = 512) -> Path:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    store = collect(cfg)

    if store.latents_path.exists() and not force:
        log.info("latents already present -> %s", store.latents_path)
        return store.latents_path

    vae = load_vae(cfg, device)
    obs = store.open_obs("r")
    n, t = obs.shape[0], obs.shape[1]
    d = cfg.vision.latent_dim

    mu_all = np.zeros((n, t, d), dtype=np.float32)
    logvar_all = np.zeros((n, t, d), dtype=np.float32)

    # Encode one rollout at a time: a whole rollout of 300 frames is 3.7 MB, so
    # peak RAM stays trivial regardless of dataset size.
    for i in range(n):
        frames = np.asarray(obs[i], dtype=np.uint8)
        for s in range(0, t, batch_size):
            chunk = torch.from_numpy(frames[s:s + batch_size]).to(device)
            chunk = chunk.permute(0, 3, 1, 2).float().div_(255.0)
            mu, logvar = vae.encode(chunk)
            mu_all[i, s:s + chunk.shape[0]] = mu.cpu().numpy()
            logvar_all[i, s:s + chunk.shape[0]] = logvar.cpu().numpy()
        if (i + 1) % 50 == 0 or i + 1 == n:
            log.info("encoded %d/%d rollouts", i + 1, n)

    np.savez(store.latents_path, mu=mu_all, logvar=logvar_all,
             vae_checkpoint=str(vae_checkpoint_path(cfg)))
    log.info("wrote %s  (mu %s, %.1f MB)", store.latents_path, mu_all.shape,
             (mu_all.nbytes + logvar_all.nbytes) / 1e6)
    _report_latent_health(mu_all, logvar_all, store)
    return store.latents_path


def _report_latent_health(mu: np.ndarray, logvar: np.ndarray, store: RolloutStore) -> None:
    """Sanity checks that catch the two failure modes that silently ruin M training.

    1. *Posterior collapse* -- if most dimensions have ~zero variance across the
       dataset, V has discarded the cue and no memory layer can recover it.
    2. *Cue decodability* -- if a linear readout of z cannot recover the cue on
       the frames where the cue is actually on screen, the pipeline is broken
       upstream of M and any memory result would be meaningless.
    """
    std = mu.reshape(-1, mu.shape[-1]).std(axis=0)
    active = int((std > 0.1).sum())
    log.info("latent health: %d/%d dims with std>0.1 | mean sigma %.3f",
             active, mu.shape[-1], float(np.exp(0.5 * logvar).mean()))

    meta = store.load_meta()
    cue, phase = meta["cue"], meta["phase"]
    on_screen = (phase == 0) & (cue >= 0)
    if on_screen.sum() > 100:
        acc = _linear_probe_accuracy(mu[on_screen], cue[on_screen])
        log.info("cue linearly decodable from z while on screen: %.1f%% "
                 "(must be near 100%% or the benchmark is measuring nothing)", 100 * acc)


def _linear_probe_accuracy(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> float:
    """Closed-form multinomial ridge probe; no torch, no iteration."""
    classes = np.unique(y)
    if classes.size < 2:
        return float("nan")
    n = min(len(x), 20000)
    idx = np.random.default_rng(0).choice(len(x), n, replace=False)
    x, y = x[idx], y[idx]
    split = int(0.8 * n)
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
    onehot = (y[:, None] == classes[None, :]).astype(np.float32)

    xtr, ytr = xb[:split], onehot[:split]
    w = np.linalg.solve(xtr.T @ xtr + ridge * np.eye(xb.shape[1], dtype=np.float32),
                        xtr.T @ ytr)
    pred = classes[np.argmax(xb[split:] @ w, axis=1)]
    return float((pred == y[split:]).mean())
