"""Causal self-attention as a memory core.

Lives in ``wmcore`` rather than in a model folder because two different model
folders need it -- ``World Model/`` registers it as the attention *control*, and
``Supreme/`` registers the paper's ``Hope-Attention`` variant on top of it.
Duplicating it would let the two drift, which is the one thing this repository's
layout exists to prevent.  It is a shared primitive, like the MDN head.

Why a memory study needs this baseline
--------------------------------------
Behrouz et al. (2025) frame softmax attention as the limiting case of their own
analysis: a "perfect memory" that caches every past token and has an update
frequency of infinity (their Section 8).  It is, in other words, the strongest
possible retention baseline -- nothing is ever forgotten, because nothing is
ever compressed.  A paper claiming that a learned memory layer remembers better
than an LSTM has not said much until it is placed next to attention, and the
nested-learning paper itself benchmarks against it.

The interesting comparison is therefore not "who remembers more" -- attention
wins that by construction over any bounded state -- but what each costs.
Attention's state grows linearly with the episode and its compute quadratically;
Titans' state is a fixed set of matrices.  If Hope matches attention's retention
at constant state size, that is the result.  ``wmcore.bench.cost`` measures
exactly this, and ``max_context`` below is where the asymmetry becomes visible.

Implementation notes
--------------------
* Pre-norm, single block, multi-head, with a small feed-forward so the parameter
  count lands within ~1% of the LSTM baseline's 299,008 (see ``d_ff``).
* ``step()`` maintains an explicit KV cache so single-timestep rollout produces
  the same numbers as the batched sequence pass -- required by
  ``tests/test_core.py::test_backbone_forward_matches_step``.
* ``max_context`` bounds the cache.  ``None`` means unbounded (true attention);
  an integer turns it into a sliding window, which is the honest way to compare
  against a fixed-size recurrent state at long horizons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionState:
    """KV cache.  ``k``/``v`` are ``(B, T, H, D)`` with ``T`` the history so far."""

    k: torch.Tensor
    v: torch.Tensor
    last_out: torch.Tensor

    def detach_state(self) -> "AttentionState":
        return AttentionState(self.k.detach(), self.v.detach(), self.last_out.detach())


class CausalSelfAttention(nn.Module):
    """One pre-norm causal attention block with a small feed-forward."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 50,
                 max_context: int | None = None, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.max_context = max_context
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.norm_attn = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.proj = nn.Linear(self.d_model, self.d_model)

        self.norm_ff = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, self.d_model)
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------- state --
    def initial_state(self, batch_size: int, device: torch.device) -> AttentionState:
        empty = torch.zeros(batch_size, 0, self.n_heads, self.d_head, device=device)
        return AttentionState(k=empty, v=empty.clone(),
                              last_out=torch.zeros(batch_size, self.d_model, device=device))

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_context is not None and x.shape[1] > self.max_context:
            return x[:, -self.max_context:]
        return x

    # ------------------------------------------------------------ passes --
    def _split(self, x: torch.Tensor):
        B, T, _ = x.shape
        qkv = self.qkv(self.norm_attn(x)).view(B, T, 3, self.n_heads, self.d_head)
        return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]        # each (B, T, H, D)

    def _finish(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        B, T = attn.shape[0], attn.shape[1]
        h = x + self.drop(self.proj(attn.reshape(B, T, self.d_model)))
        return h + self.drop(self.ff(self.norm_ff(h)))

    def forward(self, x: torch.Tensor, state: AttentionState | None = None
                ) -> tuple[torch.Tensor, AttentionState]:
        B, T, _ = x.shape
        q, k, v = self._split(x)

        if state is not None and state.k.shape[1] > 0:
            k = torch.cat([state.k, k], dim=1)
            v = torch.cat([state.v, v], dim=1)
        past = k.shape[1] - T

        # (B, H, T, S) scores with a causal mask that accounts for cached history.
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        idx_q = torch.arange(T, device=x.device)[:, None] + past
        idx_k = torch.arange(k.shape[1], device=x.device)[None, :]
        mask = idx_k > idx_q
        if self.max_context is not None:
            mask = mask | (idx_q - idx_k >= self.max_context)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.einsum("bhts,bshd->bthd", scores.softmax(dim=-1), v)
        y = self._finish(x, attn)
        return y, AttentionState(k=self._trim(k), v=self._trim(v), last_out=y[:, -1])

    def step(self, x_t: torch.Tensor, state: AttentionState
             ) -> tuple[torch.Tensor, AttentionState]:
        y, new_state = self.forward(x_t.unsqueeze(1), state)
        return y.squeeze(1), new_state

    def metrics(self) -> dict[str, float]:
        return {}
