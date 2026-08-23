"""Stage 1 -- train V (the ConvVAE).

Trained **once per environment** and then frozen.  Both the baseline and
Supreme load the identical checkpoint, so the latent space they model is
literally the same space.  ``scripts/run_study.py`` enforces this; if you train
V separately per model you have introduced a second axis of variation and the
comparison no longer isolates the memory layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wmcore.config import Config
from wmcore.data.collect import collect, dataset_dir, summarise
from wmcore.data.dataset import FrameDataset
from wmcore.data.store import RolloutStore
from wmcore.utils import (
    JSONLLogger,
    Stopwatch,
    count_parameters,
    get_logger,
    pick_device,
    save_checkpoint,
    seed_everything,
)
from wmcore.utils.device import MemoryBudget, device_info
from wmcore.vision import ConvVAE, vae_loss

log = get_logger(__name__)


def vae_checkpoint_path(cfg: Config) -> Path:
    """V lives with the *dataset*, not with the model run.

    Putting it here is what makes 'share one V across both models' the path of
    least resistance rather than something you have to remember to do.
    """
    if cfg.vision.checkpoint:
        return Path(cfg.vision.checkpoint)
    return dataset_dir(cfg) / f"vae_z{cfg.vision.latent_dim}_s{cfg.seed}.pt"


def train_vae(cfg: Config, *, force: bool = False) -> Path:
    ckpt_path = vae_checkpoint_path(cfg)
    if ckpt_path.exists() and not force:
        log.info("V already trained -> %s (use --force to retrain)", ckpt_path)
        return ckpt_path

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    budget = MemoryBudget(cfg.memory_budget_mb, label="train_vae")
    log.info("device: %s", device_info(device))

    store = collect(cfg)
    log.info("dataset: %s", summarise(store))

    train_idx, val_idx = store.split_indices(cfg.data.val_fraction, seed=cfg.seed)
    train_ds = FrameDataset(store, train_idx, max_frames=cfg.data.frames_in_ram, seed=cfg.seed)
    val_ds = FrameDataset(store, val_idx, max_frames=cfg.data.frames_in_ram // 5, seed=cfg.seed)
    log.info("frames: train=%d val=%d", len(train_ds), len(val_ds))

    # num_workers=2: enough to hide memmap page-faults, few enough that we do
    # not pay for two extra spawned interpreters on a 10-core laptop.
    loader_kw = dict(batch_size=cfg.vision.batch_size, num_workers=2,
                     pin_memory=False, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    # drop_last=False on validation: with a small dataset the val split can be
    # smaller than one batch, and silently evaluating on zero batches is the
    # kind of bug that produces a confident, meaningless number.
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)
    if len(val_ds) == 0:
        raise ValueError("empty validation split; raise data.n_rollouts or data.val_fraction")

    model = ConvVAE(latent_dim=cfg.vision.latent_dim,
                    channels=tuple(cfg.vision.channels),
                    image_size=cfg.env.image_size).to(device)
    log.info("V parameters: %s", f"{count_parameters(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.vision.lr,
                           weight_decay=cfg.vision.weight_decay)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch("vae")

    with JSONLLogger(run_dir / "metrics.jsonl", extra={"stage": "vae", "seed": cfg.seed}) as jlog:
        step = 0
        for epoch in range(cfg.vision.epochs):
            model.train()
            agg = _Agg()
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                out = model(batch)
                losses = vae_loss(out, batch, beta=cfg.vision.beta,
                                  free_bits=cfg.vision.free_bits,
                                  recon=cfg.vision.recon_loss)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.vision.grad_clip)
                opt.step()
                agg.add(losses)
                step += 1

            val = _evaluate(model, val_loader, cfg, device)
            watch.lap(f"epoch{epoch}")
            jlog.write(epoch=epoch, step=step, split="train", **agg.mean())
            jlog.write(epoch=epoch, step=step, split="val", **val)
            log.info("epoch %2d | train loss %8.2f | val loss %8.2f | recon %7.2f | "
                     "kl %6.2f | active dims %4.1f | rss %.0f MB",
                     epoch, agg.mean()["loss"], val["loss"], val["recon"],
                     val["kl"], val["active_dims"], budget.check())

    save_checkpoint(ckpt_path, model, optimizer=opt, epoch=cfg.vision.epochs,
                    config=cfg.to_dict(),
                    extra={"timing": watch.summary(),
                           "params": count_parameters(model),
                           "device": device_info(device)})
    _save_reconstructions(model, val_ds, device, run_dir / "vae_reconstructions.png")
    return ckpt_path


@torch.no_grad()
def _evaluate(model, loader, cfg: Config, device) -> dict:
    model.eval()
    agg = _Agg()
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        agg.add(vae_loss(out, batch, beta=cfg.vision.beta,
                         free_bits=cfg.vision.free_bits, recon=cfg.vision.recon_loss))
    return agg.mean()


class _Agg:
    """Running mean of a dict of scalar tensors."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def add(self, losses: dict) -> None:
        for k, v in losses.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.detach().item())
        self.n += 1

    def mean(self) -> dict[str, float]:
        if self.n == 0:
            raise RuntimeError("no batches were aggregated -- empty DataLoader?")
        return {k: v / self.n for k, v in self.sums.items()}


@torch.no_grad()
def _save_reconstructions(model, dataset, device, path: Path, n: int = 8) -> None:
    """Qualitative check.  If the cue colour is not legible in the reconstruction,
    the memory experiment is measuring nothing -- V has thrown the signal away."""
    try:
        from PIL import Image
    except ImportError:
        return
    model.eval()
    idx = np.linspace(0, len(dataset) - 1, n).astype(int)
    x = torch.stack([dataset[int(i)] for i in idx]).to(device)
    recon = model(x).recon
    grid = torch.cat([x, recon], dim=0).cpu().permute(0, 2, 3, 1).numpy()
    grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    rows = [np.concatenate(list(grid[i * n:(i + 1) * n]), axis=1) for i in range(2)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    log.info("wrote %s (top: input, bottom: reconstruction)", path)


def load_vae(cfg: Config, device) -> ConvVAE:
    from wmcore.utils import load_checkpoint

    model = ConvVAE(latent_dim=cfg.vision.latent_dim,
                    channels=tuple(cfg.vision.channels),
                    image_size=cfg.env.image_size).to(device)
    load_checkpoint(vae_checkpoint_path(cfg), model, map_location=str(device))
    return model.eval()
