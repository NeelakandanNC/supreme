"""The benchmark suite: run everything, write one JSON per (model, seed).

``bench.json`` is the only artefact the reporting layer reads.  That separation
means a results table can be regenerated in a second, and that adding a metric
never requires retraining anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wmcore.bench.cost import cost_report
from wmcore.bench.decision import decision_accuracy
from wmcore.bench.imagination import open_loop_curve, teacher_forced_metrics
from wmcore.bench.probes import collect_features, retention_curve
from wmcore.config import Config
from wmcore.data.collect import collect, summarise
from wmcore.train.encode_latents import encode_latents
from wmcore.train.train_memory import build_dataloaders, load_memory
from wmcore.utils import get_logger, pick_device, seed_everything
from wmcore.utils.device import device_info

log = get_logger(__name__)


def bench_path(cfg: Config) -> Path:
    return cfg.run_dir / "bench.json"


def run_benchmarks(cfg: Config, *, include_control: bool = True,
                   include_continual: bool = False) -> dict:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    store = collect(cfg)
    encode_latents(cfg)
    model = load_memory(cfg, device)

    _, val_loader, ds = build_dataloaders(cfg, store)
    _, long_loader, _ = build_dataloaders(cfg, store, seq_len=cfg.data.eval_sequence_length)

    results: dict = {
        "meta": {
            "name": cfg.name,
            "backbone": cfg.memory.backbone,
            "backbone_kwargs": cfg.memory.backbone_kwargs,
            "seed": cfg.seed,
            "env": cfg.env.id,
            "env_kwargs": cfg.env.kwargs,
            "train_seq_len": cfg.data.sequence_length,
            "eval_seq_len": cfg.data.eval_sequence_length,
            "device": device_info(device),
            "dataset": summarise(store),
        }
    }

    log.info("[1/6] decision accuracy (did M learn the rule?)")
    results["decision"] = decision_accuracy(model, long_loader, device)

    log.info("[2/6] teacher-forced prediction")
    results["teacher_forced"] = teacher_forced_metrics(model, val_loader, device)

    # Length extrapolation is only meaningful when the evaluation window is
    # longer than the training window.  On the delayed-recall environments it
    # cannot be: a training window has to span cue -> decision, which is the
    # whole episode.  There, extrapolation is tested by evaluating on a LONGER
    # CORRIDOR (see scripts/sweep_horizon.py), not on longer windows -- so the
    # section records that rather than reporting a duplicate of the line above.
    if cfg.data.eval_sequence_length > cfg.data.sequence_length:
        log.info("[3/6] length extrapolation (train %d -> eval %d)",
                 cfg.data.sequence_length, cfg.data.eval_sequence_length)
        results["length_extrapolation"] = teacher_forced_metrics(
            model, long_loader, device, max_batches=16)
    else:
        log.info("[3/6] length extrapolation: not applicable "
                 "(eval window == train window); use the corridor sweep instead")
        results["length_extrapolation"] = {
            "not_applicable": True,
            "reason": "training windows must span the full episode on this "
                      "environment; vary corridor_length to test extrapolation",
        }

    log.info("[4/6] open-loop imagination")
    results["imagination"] = open_loop_curve(
        model, long_loader, device, horizons=tuple(cfg.bench.horizons),
        context=min(32, cfg.data.sequence_length // 2), sample=False)
    results["imagination_sampled"] = open_loop_curve(
        model, long_loader, device, horizons=tuple(cfg.bench.horizons),
        context=min(32, cfg.data.sequence_length // 2), sample=True,
        temperature=cfg.memory.temperature, max_batches=4)

    log.info("[5/6] memory retention probes")
    feats = collect_features(model, long_loader, device)
    results["probes"] = retention_curve(feats, tuple(cfg.bench.probe_delays), seed=cfg.seed)

    log.info("[6/6] cost")
    results["cost"] = cost_report(model, device, latent_dim=ds.latent_dim,
                                  action_dim=ds.action_dim,
                                  seq_len=cfg.data.sequence_length,
                                  batch_size=cfg.memory.batch_size)

    if include_control:
        from wmcore.train.train_controller import controller_path, evaluate_controller

        if controller_path(cfg).exists():
            log.info("[+] control evaluation in the real environment")
            results["control"] = evaluate_controller(cfg)
        else:
            log.info("no controller found; skipping control evaluation")

    if include_continual:
        from wmcore.bench.continual import run_continual

        log.info("[+] continual learning")
        results["continual"] = run_continual(cfg)

    out = bench_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
    log.info("wrote %s", out)
    return results


def _default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
