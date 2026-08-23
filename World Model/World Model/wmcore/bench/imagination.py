"""Benchmark 1 -- how far into the future can M dream before it drifts?

Protocol
--------
For each validation window: feed ``context`` steps of *ground-truth* latents to
warm the state, then roll ``horizon`` steps open-loop, feeding the model its own
prediction back as the next input while using the *true* action sequence.  At
each step compare against the true latent.

This is the standard world-model fidelity curve, and it is the right headline
metric for a memory paper because the two failure modes it separates are exactly
the ones a memory layer governs:

* error at small k measures local dynamics (both models should tie);
* the *slope* of the curve measures whether the state retains enough to stay on
  the true trajectory (this is where a better memory shows up).

We report both the closed-form teacher-forced NLL and the open-loop MSE, plus
NLL restricted to the environment-flagged ``memory_critical`` transitions.
"""
from __future__ import annotations

import numpy as np
import torch

from wmcore.memory.base import mdn_mean, mdn_nll, mdn_sample


def _next_latent(logpi: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor,
                 *, sample: bool, temperature: float) -> torch.Tensor:
    """The latent fed back into the model during an open-loop rollout.

    ``sample=False`` uses the mixture mean: deterministic and low-variance, which
    is what you want when comparing two models over a handful of seeds.
    ``sample=True`` draws from the temperature-tau mixture, i.e. reproduces the
    dream the controller actually experiences.

    All tensors are (B, D, K); the return is (B, D).
    """
    if not sample:
        return mdn_mean(logpi, mu)
    return mdn_sample(logpi.unsqueeze(1), mu.unsqueeze(1), logsigma.unsqueeze(1),
                      temperature).squeeze(1)


@torch.no_grad()
def teacher_forced_metrics(model, loader, device, *, max_batches: int | None = None) -> dict:
    """One-step NLL / MSE with ground-truth inputs at every step."""
    model.eval()
    tot_nll = tot_mse = tot_n = 0.0
    crit_nll = crit_n = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["z"], batch["action"])
        mask = batch["mask"]
        nll = mdn_nll(out.logpi, out.mu, out.logsigma, batch["z_next"])
        mse = ((mdn_mean(out.logpi, out.mu) - batch["z_next"]) ** 2).sum(-1)

        tot_nll += float((nll * mask).sum())
        tot_mse += float((mse * mask).sum())
        tot_n += float(mask.sum())

        crit = batch.get("memory_critical")
        if crit is not None:
            cm = crit * mask
            crit_nll += float((nll * cm).sum())
            crit_n += float(cm.sum())

    out = {"nll_1step": tot_nll / max(1.0, tot_n),
           "mse_1step": tot_mse / max(1.0, tot_n),
           "n_steps": int(tot_n)}
    if crit_n > 0:
        out["nll_memory_critical"] = crit_nll / crit_n
        out["n_memory_critical"] = int(crit_n)
    return out


@torch.no_grad()
def open_loop_curve(model, loader, device, *, horizons: tuple[int, ...],
                    context: int = 32, max_batches: int | None = 8,
                    sample: bool = False, temperature: float = 1.0) -> dict:
    """Open-loop error as a function of imagination horizon.

    ``sample=False`` (default) feeds back the mixture mean, giving a
    low-variance, deterministic curve -- the right choice when comparing two
    models with a handful of seeds.  ``sample=True`` reproduces the stochastic
    dream that the controller actually experiences.
    """
    model.eval()
    max_h = max(horizons)
    err_sum = np.zeros(max_h, dtype=np.float64)
    nll_sum = np.zeros(max_h, dtype=np.float64)
    counts = np.zeros(max_h, dtype=np.float64)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        z, a, mask = batch["z"], batch["action"], batch["mask"]
        B, L, _ = z.shape
        if L <= context + 1:
            continue

        # Warm-up on ground truth: consumes z_0..z_{context-1}, so the final
        # warm-up output is already the model's prediction of z_context.
        state = model.initial_state(B, device)
        out = model(z[:, :context], a[:, :context], state)
        state = out.state

        # THE open-loop hand-off.  From here the model must live on its own
        # predictions; feeding a ground-truth latent in here (even the last
        # observed one) would make the first step secretly teacher-forced and
        # would shift the whole curve one step to the right.
        z_hat = _next_latent(out.logpi[:, -1], out.mu[:, -1], out.logsigma[:, -1],
                             sample=sample, temperature=temperature)

        steps = min(max_h, L - context)
        for k in range(steps):
            t = context + k
            step_out = model.step(z_hat, a[:, t], state)
            state = step_out.state

            target = batch["z_next"][:, t]
            nll = mdn_nll(step_out.logpi, step_out.mu, step_out.logsigma,
                          target.unsqueeze(1)).squeeze(1)
            pred = mdn_mean(step_out.logpi, step_out.mu).squeeze(1)
            m = mask[:, t]

            err_sum[k] += float((((pred - target) ** 2).sum(-1) * m).sum())
            nll_sum[k] += float((nll * m).sum())
            counts[k] += float(m.sum())

            z_hat = _next_latent(step_out.logpi[:, 0], step_out.mu[:, 0],
                                 step_out.logsigma[:, 0],
                                 sample=sample, temperature=temperature)

    valid = counts > 0
    mse = np.where(valid, err_sum / np.maximum(counts, 1), np.nan)
    nll = np.where(valid, nll_sum / np.maximum(counts, 1), np.nan)

    def at(arr: np.ndarray, h: int) -> float:
        """Error *at* step h (1-indexed), NaN if the rollout never reached it."""
        return float(arr[h - 1]) if h - 1 < len(arr) else float("nan")

    def upto(arr: np.ndarray, h: int) -> float:
        """Mean error over steps 1..h."""
        window = arr[:h]
        return float(np.nanmean(window)) if np.isfinite(window).any() else float("nan")

    return {
        "horizons": list(horizons),
        # Reported separately and named exactly, because "MSE at horizon 100"
        # meaning "mean MSE over the first 100 steps" is a real source of
        # cross-paper confusion.
        "mse_at_horizon": {int(h): at(mse, h) for h in horizons},
        "mse_mean_upto": {int(h): upto(mse, h) for h in horizons},
        "mse_step": [float(x) for x in mse],
        "nll_at_horizon": {int(h): at(nll, h) for h in horizons},
        "nll_step": [float(x) for x in nll],
        "context": context,
        "sample": sample,
        "max_valid_step": int(valid.sum()),
        # Least-squares slope of log(MSE) against step index: the exponential
        # divergence rate of the imagined trajectory.  One scalar for the table,
        # and unlike a first-vs-last ratio it uses the whole curve.
        "drift_slope": _log_slope(mse),
    }


def _log_slope(mse: np.ndarray) -> float:
    """Slope of log(MSE) vs. step, over the steps that have data."""
    finite = np.isfinite(mse) & (mse > 0)
    if finite.sum() < 3:
        return float("nan")
    x = np.arange(len(mse), dtype=np.float64)[finite]
    y = np.log(mse[finite])
    return float(np.polyfit(x, y, 1)[0])
