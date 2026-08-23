"""Checkpoint save/load.

A checkpoint carries enough metadata to reconstruct the model without the
original config file, because benchmark scripts are routinely pointed at a
directory produced weeks earlier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wmcore.utils.logging_utils import get_logger

log = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: "torch.nn.Module",
    *,
    optimizer: "torch.optim.Optimizer | None" = None,
    step: int = 0,
    epoch: int = 0,
    config: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "step": step,
        "epoch": epoch,
        "config": config or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    log.info("saved checkpoint -> %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    model: "torch.nn.Module | None" = None,
    *,
    optimizer: "torch.optim.Optimizer | None" = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict:
    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
