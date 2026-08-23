"""Logging: one human-readable stream, one machine-readable stream.

Every experiment writes a ``metrics.jsonl`` next to its checkpoints.  The
benchmark report tooling reads only those files, so a result can always be
regenerated from artefacts without re-running training.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)


class JSONLLogger:
    """Append-only JSON-lines metric sink.

    Usage::

        with JSONLLogger(run_dir / "metrics.jsonl") as log:
            log.write(step=1, split="train", loss=1.23)
    """

    def __init__(self, path: str | Path, extra: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._extra = extra or {}
        self._t0 = time.time()

    def __enter__(self) -> "JSONLLogger":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, **record: Any) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        payload = {"wall_s": round(time.time() - self._t0, 3), **self._extra, **record}
        self._fh.write(json.dumps(payload, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def read_jsonl(path: str | Path) -> list[dict]:
    """Load a metrics.jsonl written by :class:`JSONLLogger`."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
