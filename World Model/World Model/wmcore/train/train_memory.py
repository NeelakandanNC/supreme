"""Stage 3 -- train M (the memory model).

This is the only stage whose *result* differs between the baseline and Supreme,
and every knob it touches is shared:

* same dataset directory, same frozen V, same precomputed latents;
* same optimiser, learning rate, schedule, gradient clipping and epoch count;
* same MDN head, same loss, same masking, same seeds.

The single difference is ``cfg.memory.backbone``.  ``assert_comparable`` in
:mod:`wmcore.config` is the mechanical check; this module additionally warns if
the two backbones' parameter counts drift apart, because "our layer is better"
and "our layer is bigger" are different papers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wmcore.config import Config
from wmcore.data.collect import collect
from wmcore.data.dataset import LatentSequenceDataset
from wmcore.memory import build_memory
from wmcore.train.encode_latents import encode_latents
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
from wmcore.utils.profiling import parameter_table

log = get_logger(__name__)


def memory_checkpoint_path(cfg: Config) -> Path:
    return cfg.run_dir / "memory.pt"


def check_window_spans_the_dependency(cfg: Config, store, seq_len: int) -> None:
    """Abort if a BPTT window cannot contain both the cue and the decision.

    This is the single easiest way to silently invalidate the whole study.  In a
    delayed-recall episode the cue appears at t=0 and the transition that needs
    it at t=T-3.  If ``sequence_length`` is shorter than that span, then *no*
    training window ever contains both: windows that start at 0 stop before the
    decision, and windows that reach the decision start after the cue.  The
    dependency is then not merely hard to learn, it is absent from the training
    signal entirely -- and every memory layer scores at chance, for a reason that
    has nothing to do with memory.

    The check reads the actual data rather than the config, so it also catches
    the case where the environment's episode length differs from what the config
    assumed.
    """
    meta = store.load_meta()
    crit = meta["memory_critical"]
    if not crit.any():
        return  # environment has no flagged dependency (e.g. CarRacing)

    rows, cols = np.nonzero(crit)
    span = int(cols.max()) + 1          # cue is at t=0, so the span is the index + 1
    if seq_len < span:
        raise ValueError(
            f"data.sequence_length={seq_len} is shorter than the cue-to-decision "
            f"span of {span} steps in {cfg.env.id}.\n"
            "No training window would contain both the cue and the transition "
            "that depends on it, so the dependency is absent from the training "
            "signal and every memory layer will score at chance.\n"
            f"Set data.sequence_length >= {span} (typically the full episode "
            "length), or shorten the corridor."
        )


def build_dataloaders(cfg: Config, store, *, seq_len: int | None = None):
    seq_len = seq_len or cfg.data.sequence_length
    check_window_spans_the_dependency(cfg, store, seq_len)
    train_idx, val_idx = store.split_indices(cfg.data.val_fraction, seed=cfg.seed)
    train_ds = LatentSequenceDataset(store, train_idx, seq_len,
                                     sample_latents=True, seed=cfg.seed)
    # Validation uses the posterior *mean*: sampling there would add variance
    # that is identical in expectation for both models but noisy per run.
    val_ds = LatentSequenceDataset(store, val_idx, seq_len,
                                   sample_latents=False, seed=cfg.seed + 1)
    kw = dict(batch_size=cfg.memory.batch_size, num_workers=0, drop_last=False)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            train_ds)


def train_memory(cfg: Config, *, force: bool = False) -> Path:
    ckpt_path = memory_checkpoint_path(cfg)
    if ckpt_path.exists() and not force:
        log.info("M already trained -> %s (use --force to retrain)", ckpt_path)
        return ckpt_path

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    budget = MemoryBudget(cfg.memory_budget_mb, label="train_memory")

    store = collect(cfg)
    encode_latents(cfg)
    train_loader, val_loader, train_ds = build_dataloaders(cfg, store)

    model = build_memory(cfg.memory, train_ds.latent_dim, train_ds.action_dim).to(device)
    table = parameter_table(model)
    log.info("M backbone=%s | params %s | breakdown %s",
             cfg.memory.backbone, f"{table['__total__']:,}",
             {k: f"{v:,}" for k, v in table.items() if k != "__total__"})
    log.info("windows: train=%d val=%d (seq_len=%d)",
             len(train_loader.dataset), len(val_loader.dataset), cfg.data.sequence_length)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.memory.lr,
                           weight_decay=cfg.memory.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.memory.epochs))

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch("memory")
    best_val = float("inf")

    with JSONLLogger(run_dir / "metrics.jsonl",
                     extra={"stage": "memory", "seed": cfg.seed,
                            "backbone": cfg.memory.backbone}) as jlog:
        step = 0
        for epoch in range(cfg.memory.epochs):
            model.train()
            agg = _Agg()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["z"], batch["action"])
                losses = model.loss(out, batch,
                                    reward_weight=cfg.memory.reward_loss_weight,
                                    done_weight=cfg.memory.done_loss_weight,
                                    critical_weight=cfg.memory.critical_loss_weight)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.memory.grad_clip)
                # Same call for every backbone; a no-op unless the memory layer
                # defines its own update schedule (see MemoryBackbone).
                model.backbone.pre_optimizer_step(step)
                opt.step()
                agg.add(losses)
                step += 1
            sched.step()

            val = evaluate_memory(model, val_loader, cfg, device)
            watch.lap(f"epoch{epoch}")
            extra = model.backbone.extra_metrics()
            jlog.write(epoch=epoch, step=step, split="train", lr=sched.get_last_lr()[0],
                       **agg.mean(), **extra)
            jlog.write(epoch=epoch, step=step, split="val", **val)
            log.info("epoch %2d | train nll %8.3f | val nll %8.3f | "
                     "val nll@memory-critical %8.3f | rss %.0f MB",
                     epoch, agg.mean()["nll_z"], val["nll_z"],
                     val.get("nll_memory_critical", float("nan")), budget.check())

            if val["nll_z"] < best_val:
                best_val = val["nll_z"]
                save_checkpoint(ckpt_path, model, optimizer=opt, epoch=epoch, step=step,
                                config=cfg.to_dict(),
                                extra={"val": val, "params": table,
                                       "latent_dim": train_ds.latent_dim,
                                       "action_dim": train_ds.action_dim,
                                       "timing": watch.summary(),
                                       "device": device_info(device)})
    log.info("best val NLL %.4f | total %.1f s", best_val, watch.elapsed)
    return ckpt_path


@torch.no_grad()
def evaluate_memory(model, loader, cfg: Config, device) -> dict:
    model.eval()
    agg = _Agg()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["z"], batch["action"])
        # Evaluation deliberately uses weight 1.0: the reported NLL must be the
        # true per-step average, not the training objective.
        agg.add(model.loss(out, batch,
                           reward_weight=cfg.memory.reward_loss_weight,
                           done_weight=cfg.memory.done_loss_weight))
    return agg.mean()


def load_memory(cfg: Config, device, path: Path | None = None):
    from wmcore.utils import load_checkpoint

    path = path or memory_checkpoint_path(cfg)
    payload = load_checkpoint(path, map_location=str(device))
    extra = payload.get("extra", {})
    model = build_memory(cfg.memory, extra["latent_dim"], extra["action_dim"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()


class _Agg:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def add(self, losses: dict) -> None:
        for k, v in losses.items():
            val = float(v.detach().item()) if torch.is_tensor(v) else float(v)
            if not np.isnan(val):
                self.sums[k] = self.sums.get(k, 0.0) + val
        self.n += 1

    def mean(self) -> dict[str, float]:
        return {k: v / max(1, self.n) for k, v in self.sums.items()}
