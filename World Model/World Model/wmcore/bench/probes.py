"""Benchmark 2 -- what is still *in* the memory state, and for how long?

A world model can have a good average NLL while having forgotten the one fact
the task depends on, because that fact costs almost nothing in average
likelihood.  Probes measure retention directly.

Method
------
Run M teacher-forced over validation sequences and harvest the recurrent
features ``h_t``.  For each cue-to-probe delay ``d``, take every timestep whose
``delay_since_cue == d`` and fit a linear (ridge) classifier from ``h_t`` to the
ground-truth cue, train/test split by episode.  Accuracy versus ``d`` is the
retention curve.

Two controls make the curve interpretable, and both are reported:

``probe_z``
    the same classifier fitted on the *latent* ``z_t`` alone.  This is what is
    visible in the current frame.  Once the cue leaves the screen this collapses
    to chance; anything above chance in ``probe_h`` past that point is memory.
``chance``
    1 / n_classes, from the observed label distribution.

Linear probes, not MLPs, on purpose: a nonlinear probe can manufacture accuracy
from a state that no linear controller could use, and C in this study *is*
linear.  The probe therefore measures exactly the thing that matters downstream.
"""
from __future__ import annotations

import numpy as np
import torch

#: Minimum number of (episode, delay) samples before a probe is considered
#: informative.  Below this the train/test split is too small to mean anything.
MIN_PROBE_SAMPLES = 24


@torch.no_grad()
def collect_features(model, loader, device, *, max_batches: int | None = 16) -> dict[str, np.ndarray]:
    """Harvest ``(h_t, z_t, cue_t, delay_t, episode_id)`` over the validation set.

    Timing note: ``h_t`` here is the backbone state *after* consuming
    ``(z_t, a_t)`` -- the same vector the MDN head sees when it predicts
    ``z_{t+1}``.  The controller instead receives the state *before* consuming
    ``z_t`` (see :mod:`wmcore.controller.agent`).  The two differ by one step;
    the post-consumption state is the right object to probe because it is the
    one the model's own prediction is conditioned on, and the definition is
    identical for every backbone, so the comparison is unaffected.
    """
    model.eval()
    H, Z, C, D, E = [], [], [], [], []
    ep = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["z"], batch["action"])
        B, L, _ = batch["z"].shape
        eps = np.arange(ep, ep + B)[:, None].repeat(L, axis=1)
        ep += B

        H.append(out.features.reshape(B * L, -1).cpu().numpy())
        Z.append(batch["z"].reshape(B * L, -1).cpu().numpy())
        C.append(batch["cue"].reshape(-1).cpu().numpy())
        D.append(batch["delay"].reshape(-1).cpu().numpy())
        E.append(eps.reshape(-1))

    return {
        "h": np.concatenate(H).astype(np.float32),
        "z": np.concatenate(Z).astype(np.float32),
        "cue": np.concatenate(C).astype(np.int64),
        "delay": np.concatenate(D).astype(np.int64),
        "episode": np.concatenate(E).astype(np.int64),
    }


def ridge_probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                *, ridge: float = 1.0, seed: int = 0) -> dict:
    """One-vs-rest ridge classification with a *group* (episode) split.

    Splitting by episode rather than by timestep is essential: consecutive
    timesteps within an episode share the same label and near-identical state,
    so a random split would report near-perfect accuracy for any model.
    """
    classes = np.unique(y)
    if classes.size < 2 or len(y) < MIN_PROBE_SAMPLES:
        return {"accuracy": float("nan"), "chance": float("nan"), "n": int(len(y))}

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = max(1, int(0.25 * len(uniq)))
    test_groups = set(uniq[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    if test_mask.all() or (~test_mask).all():
        return {"accuracy": float("nan"), "chance": float("nan"), "n": int(len(y))}

    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    onehot = (y[:, None] == classes[None, :]).astype(np.float32)
    xtr, ytr = xb[~test_mask], onehot[~test_mask]

    gram = xtr.T @ xtr + ridge * np.eye(xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(gram, xtr.T @ ytr)
    pred = classes[np.argmax(xb[test_mask] @ w, axis=1)]

    # Chance is the majority-class rate *on the test split*, not on the whole
    # sample: with a group split the two can differ noticeably, and comparing
    # test accuracy against a train-set baseline is not a comparison at all.
    y_test = y[test_mask]
    counts = np.bincount(y_test, minlength=int(classes.max()) + 1)
    return {
        "accuracy": float((pred == y_test).mean()),
        "chance": float(counts.max() / counts.sum()),
        "n": int(len(y)),
        "n_test": int(test_mask.sum()),
    }


def retention_curve(features: dict[str, np.ndarray], delays: tuple[int, ...],
                    *, seed: int = 0) -> dict:
    """Probe accuracy versus cue-to-probe delay, for ``h`` and for the ``z`` control."""
    valid = features["cue"] >= 0
    curve: dict[str, dict] = {"h": {}, "z": {}}
    for d in delays:
        sel = valid & (features["delay"] == d)
        if sel.sum() < MIN_PROBE_SAMPLES:
            # Each validation episode contributes one timestep per delay, so the
            # sample count here is the number of validation episodes.  If this
            # trips, raise data.n_rollouts or data.val_fraction rather than
            # lowering the threshold: an under-powered probe is worse than none.
            continue
        for key in ("h", "z"):
            curve[key][int(d)] = ridge_probe(
                features[key][sel], features["cue"][sel], features["episode"][sel], seed=seed
            )

    # Headline scalar: the largest delay at which the memory state still beats
    # both chance and the observation-only control by a clear margin.
    horizon = 0
    for d in sorted(curve["h"]):
        acc_h = curve["h"][d]["accuracy"]
        acc_z = curve["z"][d]["accuracy"]
        chance = curve["h"][d]["chance"]
        if acc_h > max(chance, acc_z) + 0.10:
            horizon = d
    return {
        "delays": [int(d) for d in sorted(curve["h"])],
        "probe_h": {str(k): v for k, v in curve["h"].items()},
        "probe_z": {str(k): v for k, v in curve["z"].items()},
        "effective_memory_horizon": horizon,
        "auc_h": _auc(curve["h"]),
        "auc_z": _auc(curve["z"]),
    }


def _auc(curve: dict) -> float:
    """Mean probe accuracy across delays -- a single number for the results table."""
    accs = [v["accuracy"] for v in curve.values() if not np.isnan(v["accuracy"])]
    return float(np.mean(accs)) if accs else float("nan")
