"""Benchmark 4 -- what the memory layer costs.

A laptop study lives or dies on this table.  Any claim of the form "our memory
layer is better" has to be read next to parameters, wall-clock training time,
per-step inference latency and peak resident memory, because the honest
alternative hypothesis is always "you spent more compute".
"""
from __future__ import annotations

import torch

from wmcore.utils.profiling import benchmark_step_time, count_parameters, parameter_table, peak_rss_mb


def _sync(device: torch.device):
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return None


@torch.no_grad()
def cost_report(model, device: torch.device, *, latent_dim: int, action_dim: int,
                seq_len: int = 128, batch_size: int = 32) -> dict:
    """Parameter counts plus measured throughput and latency."""
    model.eval()
    table = parameter_table(model)

    z1 = torch.randn(1, latent_dim, device=device)
    a1 = torch.randn(1, action_dim, device=device)
    state = model.initial_state(1, device)

    def one_step():
        model.step(z1, a1, state)

    zs = torch.randn(batch_size, seq_len, latent_dim, device=device)
    as_ = torch.randn(batch_size, seq_len, action_dim, device=device)

    def seq_pass():
        model(zs, as_)

    step_stats = benchmark_step_time(one_step, sync=_sync(device))
    seq_stats = benchmark_step_time(seq_pass, warmup=3, iters=10, sync=_sync(device))
    steps_per_call = batch_size * seq_len

    return {
        "params_total": int(table["__total__"]),
        "params_backbone": int(table.get("backbone", 0)),
        "params_head": int(table.get("head", 0)),
        "params_breakdown": {k: int(v) for k, v in table.items()},
        "inference_step_ms": round(step_stats["median_ms"], 4),
        "inference_step_p90_ms": round(step_stats["p90_ms"], 4),
        "train_pass_ms": round(seq_stats["median_ms"], 3),
        "throughput_steps_per_s": round(1e3 * steps_per_call / seq_stats["median_ms"], 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "device": str(device),
    }


def compare_parameter_budgets(models: dict[str, torch.nn.Module], tol: float = 0.10) -> dict:
    """Flag a parameter-count mismatch between the models being compared."""
    counts = {k: count_parameters(m) for k, m in models.items()}
    lo, hi = min(counts.values()), max(counts.values())
    ratio = hi / max(1, lo)
    return {
        "counts": counts,
        "ratio": round(ratio, 4),
        "matched": bool(ratio - 1.0 <= tol),
        "note": ("parameter counts are matched within tolerance"
                 if ratio - 1.0 <= tol else
                 f"MISMATCH: largest model is {100 * (ratio - 1):.1f}% bigger; "
                 "equalise hidden_dim / backbone_kwargs before reporting"),
    }
