"""Cost accounting.

For a laptop-scale study, wall-clock time, resident memory and parameter count
are first-class results, not footnotes: a memory layer that wins on NLL but
costs 4x the step time has not obviously won.  Every benchmark reports these
alongside quality metrics.
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is bytes on macOS/BSD and kilobytes on Linux.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def count_parameters(module: "torch.nn.Module", trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def parameter_table(module: "torch.nn.Module") -> dict[str, int]:
    """Per-top-level-submodule parameter counts, for the ablation table."""
    out: dict[str, int] = {}
    for name, child in module.named_children():
        out[name] = sum(p.numel() for p in child.parameters())
    out["__total__"] = sum(p.numel() for p in module.parameters())
    return out


@dataclass
class Stopwatch:
    """Accumulating timer with named laps."""

    name: str = "timer"
    laps: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)
    _mark: float = field(default_factory=time.perf_counter)

    def lap(self, label: str) -> float:
        now = time.perf_counter()
        dt = now - self._mark
        self._mark = now
        self.laps[label] = self.laps.get(label, 0.0) + dt
        return dt

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def summary(self) -> dict[str, float]:
        return {"total_s": round(self.elapsed, 3),
                **{f"{k}_s": round(v, 3) for k, v in self.laps.items()}}


def benchmark_step_time(fn, *, warmup: int = 5, iters: int = 30, sync=None) -> dict:
    """Median/mean wall-clock of a callable, used for inference-latency tables."""
    import statistics

    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": 1e3 * statistics.fmean(times),
        "median_ms": 1e3 * statistics.median(times),
        "p90_ms": 1e3 * sorted(times)[int(0.9 * len(times)) - 1],
        "iters": iters,
    }
