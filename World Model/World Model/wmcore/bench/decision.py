"""Benchmark 0 -- did the world model actually learn the rule?

NLL is a poor instrument for the question this study asks.  On the synthetic
memory environments the one transition that requires memory is a single step out
of ~30, so its contribution to average NLL is diluted roughly thirty-fold; and a
model can lower NLL on that step just by hedging between the two possible
outcomes without ever having learned which one follows.

This metric asks the question directly, in the model's own output space:

    at the decision step, does M predict the frame that *actually* comes next?

The two possible successors (a green "correct" frame and a red "wrong" frame)
occupy two well-separated regions of latent space.  We estimate their centroids
from the ground-truth data, take the model's predicted next latent, assign it to
the nearer centroid, and check it against the truth.

Properties that make this the right headline number:

* **Chance is exactly 1/2** (a uniformly random data-collection policy is right
  half the time), so the scale is interpretable without a baseline model.
* **It cannot be gamed by hedging.**  A model that predicts the average of the
  two outcomes lands between the centroids and scores at chance.
* **It requires memory by construction.**  The outcome is a function of the cue
  (seen at t=0) and the action (seen now); nothing in the current frame reveals it.
* **It is measured on the model, not on a controller**, so it carries none of the
  variance that makes CMA-ES results hard to read.
"""
from __future__ import annotations

import numpy as np
import torch

from wmcore.memory.base import mdn_mean


@torch.no_grad()
def decision_accuracy(model, loader, device, *, max_batches: int | None = None) -> dict:
    """Nearest-centroid accuracy of the predicted post-decision frame.

    Returns ``accuracy`` (chance = 0.5), ``n`` and ``margin`` -- the mean
    normalised distance between the two centroids, which is a sanity check that
    the two outcomes really are separable in latent space.  If ``margin`` is
    small the metric is meaningless and the VAE, not the memory layer, is the
    problem.
    """
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        crit = (batch["memory_critical"] * batch["mask"]).bool()
        if not crit.any():
            continue

        out = model(batch["z"], batch["action"])
        pred = mdn_mean(out.logpi, out.mu)              # (B, L, D)

        # The environment rewards `correct_action`; the agent took argmax(action).
        taken = batch["action"].argmax(dim=-1)          # (B, L)
        was_correct = (taken == batch["correct_action"]).float()

        preds.append(pred[crit].cpu().numpy())
        truths.append(batch["z_next"][crit].cpu().numpy())
        labels.append(was_correct[crit].cpu().numpy())

    if not preds:
        return {"accuracy": float("nan"), "n": 0, "margin": float("nan")}

    pred = np.concatenate(preds)
    truth = np.concatenate(truths)
    label = np.concatenate(labels).astype(int)

    if len(np.unique(label)) < 2:
        return {"accuracy": float("nan"), "n": int(len(label)),
                "margin": float("nan"),
                "note": "only one outcome present; need a mix of right and wrong answers"}

    # Centroids come from the ground-truth successors, never from the model, so
    # a bad model cannot move the decision boundary in its own favour.
    c_wrong = truth[label == 0].mean(axis=0)
    c_right = truth[label == 1].mean(axis=0)
    sep = float(np.linalg.norm(c_right - c_wrong))

    d_wrong = np.linalg.norm(pred - c_wrong, axis=1)
    d_right = np.linalg.norm(pred - c_right, axis=1)
    assigned = (d_right < d_wrong).astype(int)

    scale = float(np.linalg.norm(truth - truth.mean(axis=0), axis=1).mean())
    return {
        "accuracy": float((assigned == label).mean()),
        "n": int(len(label)),
        "margin": sep / max(scale, 1e-8),
        "chance": 0.5,
        "positive_rate": float(label.mean()),
    }
