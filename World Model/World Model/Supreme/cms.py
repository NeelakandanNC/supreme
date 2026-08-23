"""Continuum Memory System (CMS).

Implements Section 7.1 of Behrouz et al. (NeurIPS 2025), Equations 70-71: a
chain of MLP blocks in which each block is updated at a **different frequency**::

    y_t = MLP^(f_k)( MLP^(f_{k-1})( ... MLP^(f_1)(x_t) ) )                (Eq. 70)

    theta^(f_l)_{i+1} = theta^(f_l)_i - sum_{t=i-C^(l)}^{i} eta ∇L(theta_t; x_t)
                                                      if i = 0 (mod C^(l))    (Eq. 71)
    theta^(f_l)_{i+1} = theta^(f_l)_i                 otherwise

The paper's argument (Section 7.1, "CMS Design Helps with Continual Learning"):
when a fast block is overwritten by new data, the knowledge it held is still
present in the slower blocks, and back-propagation can circulate it back --
so the system forgets reluctantly. A conventional Transformer block is the
degenerate case ``k = 1`` with update frequency zero.

How Equation 71 is read here
----------------------------
Equation 71 is explicit that ``L`` is "the objective of choice for the task at
hand, e.g., for language modeling it is next token prediction objective" -- the
*outer* task loss, not a surrogate internal to the layer. Read literally, it
therefore describes a **gradient-accumulation schedule over optimiser steps**:
block ``l`` accumulates the task gradient for ``C^(l)`` steps and applies the sum
once. That is what is implemented, exactly, in :meth:`pre_optimizer_step`.

This is worth stating plainly because the paper also gestures at a
within-sequence reading (chunk sizes ``C^(l) = L / f``, "sequence
parallelization for higher frequency levels"). The two readings are not
equivalent. We take the literal one because it is unambiguous, exactly
implementable, and matches both the M3 optimiser of Section 7.2 -- which applies
the same idea to momentum across optimiser steps -- and Section 7.3's proposal
to initialise CMS blocks from pre-trained MLP weights.

The variant implemented is the **sequential** one (Eq. 73): blocks are chained,
and their initial states are all learned by back-propagation at the lowest
frequency. Equation 74's independent/head-wise variant is available via
``aggregate=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CMSBlock(nn.Module):
    """One level of the chain: a pre-norm residual MLP with its own period."""

    def __init__(self, d_model: int, d_ff: int, period: int, dropout: float = 0.0):
        super().__init__()
        self.period = max(1, int(period))
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class ContinuumMemorySystem(nn.Module):
    """The chain of Equation 70, with the update schedule of Equation 71.

    Parameters
    ----------
    d_model:
        Width of the stream.
    d_ff:
        Hidden width inside each block.
    periods:
        ``C^(l)`` per level, fastest first.  ``(1, 4, 16)`` means level 1 updates
        every optimiser step, level 2 every 4, level 3 every 16 -- a spectrum
        from working memory to slowly-consolidated knowledge.
    aggregate:
        ``False`` (default) chains the blocks, Eq. 70/73.  ``True`` runs them in
        parallel on the same input and combines with a learned weighted sum,
        Eq. 74's independent variant.
    """

    def __init__(self, d_model: int = 256, d_ff: int = 64,
                 periods: tuple[int, ...] = (1, 4, 16), aggregate: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.periods = tuple(int(p) for p in periods)
        self.aggregate = bool(aggregate)
        self.blocks = nn.ModuleList(
            [CMSBlock(d_model, d_ff, p, dropout) for p in self.periods]
        )
        if self.aggregate:
            # Learned weighted sum, the simple choice the paper suggests for Agg.
            self.mix = nn.Parameter(torch.zeros(len(self.blocks)))

        # Gradient accumulators for the levels that update less than every step.
        self._accum: dict[str, torch.Tensor] = {}
        self._applied = [0 for _ in self.blocks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.aggregate:
            for block in self.blocks:
                x = block(x)
            return x
        weights = torch.softmax(self.mix, dim=0)
        return sum(w * block(x) for w, block in zip(weights, self.blocks))

    # ------------------------------------------------------- Equation 71 --
    @torch.no_grad()
    def pre_optimizer_step(self, global_step: int) -> None:
        """Gate each level's gradient so it is applied only every ``C^(l)`` steps.

        Called by the training loop after ``backward()`` and before
        ``optimizer.step()``.  For a level with period ``C``:

        * on the ``C-1`` intermediate steps the gradient is moved into an
          accumulator and the parameter's ``.grad`` is zeroed, so the optimiser
          leaves it untouched;
        * on the ``C``-th step the accumulated **sum** is written back into
          ``.grad``, which is exactly the summation in Equation 71.

        Accumulating rather than dropping matters: dropping would make a slow
        level simply see less data, whereas summing makes it see the same data
        at a coarser temporal resolution -- which is the whole point.
        """
        for i, block in enumerate(self.blocks):
            if block.period <= 1:
                self._applied[i] += 1
                continue
            fire = (global_step + 1) % block.period == 0
            for name, p in block.named_parameters():
                if p.grad is None:
                    continue
                key = f"{i}.{name}"
                buf = self._accum.get(key)
                buf = p.grad.detach().clone() if buf is None else buf + p.grad.detach()
                if fire:
                    p.grad.copy_(buf)
                    self._accum.pop(key, None)
                else:
                    self._accum[key] = buf
                    p.grad.zero_()
            if fire:
                self._applied[i] += 1

    def metrics(self) -> dict[str, float]:
        return {f"cms_updates_level{i}": float(n) for i, n in enumerate(self._applied)}
