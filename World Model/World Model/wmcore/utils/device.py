"""Device selection tuned for an Apple-silicon laptop.

Target hardware for this study is an M4 MacBook with 16 GB unified memory, of
which roughly 10 GB is realistically usable.  That budget drives three
decisions baked into this file and the default configs:

1. **MPS by default.**  Apple's Metal backend gives a solid speedup over CPU
   for the conv stack (V) and for batched sequence models (M).
2. **float32 everywhere.**  MPS float16 autocast is still flaky for LSTM and
   for the custom kernels used by the nested-learning memory; the models here
   are small enough that fp32 is affordable and removes a confound.
3. **A hard memory ceiling.**  ``MemoryBudget`` lets training scripts fail
   loudly *before* the OS starts swapping, instead of silently thrashing.
"""
from __future__ import annotations

import contextlib
import os
import platform
from dataclasses import dataclass


def pick_device(preferred: str = "auto") -> "torch.device":
    """Return the best available torch device.

    ``preferred`` may be ``auto``, ``mps``, ``cuda`` or ``cpu``.
    """
    import torch

    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device: "torch.device") -> dict:
    """Human/JSON friendly description of the compute environment.

    Recorded in every run manifest: on a laptop study the hardware *is* part of
    the result, and reviewers will ask.
    """
    import torch

    info = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "mps":
        info["mps_built"] = torch.backends.mps.is_built()
    if device.type == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


@contextlib.contextmanager
def autocast_ctx(device: "torch.device", enabled: bool = False):
    """No-op autocast by default.

    Deliberately opt-in: mixed precision changes numerics, and a numerics
    change that lands on only one of the two models would invalidate the
    comparison.  Left here so it can be enabled uniformly if ever needed.
    """
    import torch

    if not enabled or device.type == "cpu":
        yield
        return
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        yield


@dataclass
class MemoryBudget:
    """Soft guard against exceeding the laptop's usable RAM.

    Not a hard allocator limit -- it is a tripwire.  Training scripts call
    :meth:`check` once per epoch; if resident memory crosses the ceiling the
    run aborts with an actionable message rather than dragging the machine
    into swap for an hour.
    """

    limit_mb: float = 9500.0  # ~10 GB usable on a 16 GB M4 with apps closed
    label: str = "run"

    def check(self, extra: str = "") -> float:
        from wmcore.utils.profiling import peak_rss_mb

        rss = peak_rss_mb()
        if rss > self.limit_mb:
            raise MemoryError(
                f"[{self.label}] resident memory {rss:.0f} MB exceeded the "
                f"budget of {self.limit_mb:.0f} MB. {extra}\n"
                "Lower `data.frames_in_ram`, `train.batch_size` or "
                "`data.sequence_length` in the config."
            )
        return rss
