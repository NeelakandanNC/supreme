"""Backbone registry.

Adding a memory layer to the study is exactly two steps:

1. drop a folder at the project root with an ``__init__.py``;
2. inside it, subclass :class:`~wmcore.memory.base.MemoryBackbone` and call
   :func:`register_backbone` with a key.

The folder is discovered automatically -- ``World Model/`` (the Ha & Schmidhuber
baseline) and ``Supreme/`` (this work) are found exactly this way, spaces and
all.

Nothing else in the repository changes -- not the data pipeline, not the
training loop, not the controller, not the benchmark.  That is the whole point
of the layout: the ablation is enforced by the architecture of the code, not by
the discipline of whoever runs the experiments.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from wmcore.memory.base import MemoryBackbone, MemoryModule
from wmcore.utils.logging_utils import get_logger

log = get_logger(__name__)

_BACKBONES: dict[str, Callable[..., MemoryBackbone]] = {}

#: Project root -- the directory holding wmcore/ and the model folders.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Top-level directories that are infrastructure, not models.
_NOT_MODELS = {"wmcore", "configs", "docs", "tests", "scripts", "runs",
               ".venv", ".git", "__pycache__"}

_loaded = False


def register_backbone(name: str, builder: Callable[..., MemoryBackbone]) -> None:
    _BACKBONES[name] = builder


def model_packages() -> list[Path]:
    """Every model folder at the project root.

    A model folder is any top-level directory with an ``__init__.py`` that is not
    infrastructure.  Discovery is by path rather than by import name so the
    folders can be called what the study calls them -- ``World Model/`` and
    ``Supreme/`` -- including the space, which is not a legal Python identifier.
    """
    out = []
    for entry in sorted(PROJECT_ROOT.iterdir()):
        if (entry.is_dir() and entry.name not in _NOT_MODELS
                and not entry.name.startswith("runs")
                and (entry / "__init__.py").exists()):
            out.append(entry)
    return out


def _module_name(directory: Path) -> str:
    """A legal module name for a folder whose real name may contain spaces."""
    return "wm_model_" + "".join(
        c if c.isalnum() else "_" for c in directory.name
    ).strip("_").lower()


def _load_package(directory: Path) -> None:
    """Import a model folder as a proper package under a sanitised name.

    ``submodule_search_locations`` is what makes the package real rather than a
    loose module: it lets ``Supreme/hope.py`` do ``from .cms import ...`` even
    though the directory it lives in could never be typed in an import
    statement.
    """
    name = _module_name(directory)
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load model package at {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _autoload() -> None:
    """Import every model folder once, so each can register its backbones."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for directory in model_packages():
        try:
            _load_package(directory)
        except Exception as exc:
            # One broken model must not take down the others -- in particular a
            # half-finished challenger must never stop the baseline from running.
            log.warning("could not load model folder %r: %s", directory.name, exc)


def available_backbones() -> list[str]:
    _autoload()
    return sorted(_BACKBONES)


def build_backbone(name: str, input_dim: int, hidden_dim: int, **kwargs: Any) -> MemoryBackbone:
    _autoload()
    if name not in _BACKBONES:
        raise KeyError(
            f"unknown memory backbone {name!r}; available: {sorted(_BACKBONES)}"
        )
    return _BACKBONES[name](input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)


def build_memory(cfg, latent_dim: int, action_dim: int) -> MemoryModule:
    """Assemble the full M model from a :class:`~wmcore.config.MemoryConfig`."""
    backbone = build_backbone(
        cfg.backbone,
        input_dim=latent_dim + action_dim,
        hidden_dim=cfg.hidden_dim,
        **cfg.backbone_kwargs,
    )
    return MemoryModule(
        backbone,
        latent_dim=latent_dim,
        action_dim=action_dim,
        n_mixtures=cfg.n_mixtures,
        predict_reward=cfg.predict_reward,
        predict_done=cfg.predict_done,
        temperature=cfg.temperature,
    )
