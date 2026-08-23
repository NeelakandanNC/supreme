"""Benchmark 3 -- continual learning and catastrophic forgetting.

Why this benchmark exists
-------------------------
Retention *within* a sequence (Benchmark 2) and retention *across tasks* are
different capabilities, and a memory layer can win one while losing the other.
Continual learning is also the claim the nested-learning line of work makes
most directly, so a paper proposing a nested-learning memory has to be measured
on it rather than only on likelihood.

Protocol (standard task-incremental setup)
------------------------------------------
Tasks ``T_0 .. T_{n-1}`` are variants of the same environment that share pixel
statistics and differ only in the cue -> correct-action mapping.  M is trained on
each task in turn, with **no replay and no task label**.  After finishing task
``i`` we evaluate on every task ``j <= i``.

Reported metrics, following Lopez-Paz & Ranzato (2017):

``ACC``          mean accuracy/NLL over all tasks after the last one
``BWT``          backward transfer: mean change on earlier tasks caused by later
                 training.  Negative BWT is forgetting.
``FWT``          forward transfer: performance on a task before ever training on
                 it, relative to chance.
``retention``    probe accuracy on T_0 after training through T_{n-1}

Keeping V frozen across tasks (``vision.checkpoint``) is deliberate: it confines
forgetting to the memory layer, which is what we are measuring.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from wmcore.config import Config
from wmcore.bench.decision import decision_accuracy
from wmcore.data.collect import collect
from wmcore.memory import build_memory
from wmcore.train.encode_latents import encode_latents
from wmcore.train.train_memory import build_dataloaders, evaluate_memory
from wmcore.train.train_vae import train_vae, vae_checkpoint_path
from wmcore.utils import get_logger, pick_device, seed_everything

log = get_logger(__name__)


def make_task_config(cfg: Config, variant: int, vae_ckpt: Path) -> Config:
    """A task = the base config with a different environment variant."""
    task = Config.from_dict(copy.deepcopy(cfg.to_dict()))
    task.env.kwargs = {**cfg.env.kwargs, "variant": variant}
    task.vision.checkpoint = str(vae_ckpt)
    # Continual runs use less data per task; the point is the sequence of tasks,
    # not the asymptote on any one of them.  With n_tasks datasets to collect and
    # encode, a quarter-size dataset per task keeps the whole protocol to a few
    # minutes instead of the better part of an hour.
    task.data.n_rollouts = max(64, cfg.data.n_rollouts // 4)
    return task


def run_continual(cfg: Config, *, n_tasks: int | None = None,
                  epochs_per_task: int | None = None) -> dict:
    n_tasks = n_tasks or cfg.bench.continual_tasks
    epochs = epochs_per_task or cfg.bench.continual_epochs_per_task

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    # One V, trained on task 0, frozen and reused for every task.
    base = Config.from_dict(copy.deepcopy(cfg.to_dict()))
    base.env.kwargs = {**cfg.env.kwargs, "variant": 0}
    vae_ckpt = train_vae(base)

    tasks = [make_task_config(cfg, i, vae_ckpt) for i in range(n_tasks)]
    loaders = []
    for i, task in enumerate(tasks):
        store = collect(task)
        encode_latents(task)
        tr, va, ds = build_dataloaders(task, store)
        loaders.append({"train": tr, "val": va, "ds": ds})
        log.info("task %d: %d train windows", i, len(tr.dataset))

    model = build_memory(cfg.memory, loaders[0]["ds"].latent_dim,
                         loaders[0]["ds"].action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.memory.lr,
                           weight_decay=cfg.memory.weight_decay)

    # R[i][j] = metric on task j after finishing training task i.
    # Row 0 (before any training) gives forward transfer.
    #
    # Two matrices are kept. NLL is the conventional one, but on these tasks it
    # is nearly blind: the variants share pixel statistics and differ only in the
    # cue -> outcome rule, which touches one transition in ~30, so a change shows
    # up in the third decimal of an average. Decision accuracy asks directly
    # whether the model still knows *this task's* rule, has chance at exactly
    # 0.5, and is what the forgetting numbers below are built on.
    R = np.full((n_tasks + 1, n_tasks), np.nan)
    A = np.full((n_tasks + 1, n_tasks), np.nan)

    def _eval_all(row: int) -> None:
        for j in range(n_tasks):
            R[row, j] = evaluate_memory(model, loaders[j]["val"], cfg, device)["nll_z"]
            A[row, j] = decision_accuracy(model, loaders[j]["val"], device)["accuracy"]

    _eval_all(0)

    global_step = 0
    for i, task in enumerate(tasks):
        for _ in range(epochs):
            model.train()
            for batch in loaders[i]["train"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["z"], batch["action"])
                losses = model.loss(out, batch,
                                    reward_weight=cfg.memory.reward_loss_weight,
                                    done_weight=cfg.memory.done_loss_weight,
                                    critical_weight=cfg.memory.critical_loss_weight)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.memory.grad_clip)
                # Same call for every backbone; a no-op unless the memory layer
                # defines its own update schedule (see MemoryBackbone).
                model.backbone.pre_optimizer_step(global_step)
                opt.step()
                global_step += 1
        _eval_all(i + 1)
        log.info("after task %d | NLL %s | decision acc %s", i,
                 np.array2string(R[i + 1], precision=3),
                 np.array2string(A[i + 1], precision=3))

    # Accuracy keeps the original Lopez-Paz & Ranzato sign convention (higher is
    # better, negative BWT means forgetting).  NLL is a loss, so its conventions
    # are mirrored and named separately to avoid confusion in the results table.
    acc_bwt = float(np.mean([A[n_tasks, j] - A[j + 1, j] for j in range(n_tasks - 1)])) \
        if n_tasks > 1 else 0.0
    acc_fwt = float(np.mean([A[0, j] - 0.5 for j in range(n_tasks)]))
    nll_bwt = float(np.mean([R[n_tasks, j] - R[j + 1, j] for j in range(n_tasks - 1)])) \
        if n_tasks > 1 else 0.0

    learned = float(np.nanmean([A[i + 1, i] for i in range(n_tasks)]))
    # Forgetting is only measurable if there was something to forget.  A model
    # that never learns any individual task produces a perfectly well-formed
    # BWT number that means nothing -- measured: an LSTM at corridor 16 cannot
    # solve the single-task version at all, and its continual matrix is just its
    # output bias reflected across the variants.  Flag it loudly rather than
    # emitting a publishable-looking figure.
    informative = learned > 0.65
    note = (None if informative else
            f"UNINFORMATIVE: mean accuracy on each task immediately after "
            f"training it is {learned:.3f} (chance 0.5). The model never learned "
            f"the tasks, so the forgetting metrics are meaningless. Run continual "
            f"learning at a corridor length where this memory layer does solve "
            f"the single-task version, or with a layer that can.")
    if note:
        log.warning(note)

    return {
        "n_tasks": n_tasks,
        "epochs_per_task": epochs,
        "informative": informative,
        "note": note,
        "matrix_nll": R.tolist(),
        "matrix_decision_accuracy": A.tolist(),
        # --- primary (decision accuracy, chance = 0.5) ---
        "final_accuracy_mean": float(np.nanmean(A[n_tasks])),
        "learned_accuracy_mean": learned,
        "backward_transfer_accuracy": acc_bwt,        # < 0 means forgetting
        "forward_transfer_accuracy": acc_fwt,
        # Positive value = accuracy on task 0 dropped after later training.
        "forgetting_task0_accuracy": float(A[1, 0] - A[n_tasks, 0]),
        # --- secondary (NLL) ---
        "final_nll_mean": float(np.mean(R[n_tasks])),
        "backward_transfer_nll": nll_bwt,             # > 0 means forgetting
        "forgetting_task0_nll": float(R[n_tasks, 0] - R[1, 0]),
    }
