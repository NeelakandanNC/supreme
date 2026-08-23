"""Attention control.

Not from Ha & Schmidhuber -- it sits in this folder because, like the GRU here,
its role is to be a *control* the challenger has to beat, not a model under
test.

Behrouz et al. (2025, Section 8) characterise softmax attention as a "perfect
memory": a non-parametric solution that caches every past token and never
compresses, with an effective update frequency of infinity. It is therefore the
strongest retention baseline available, and a memory paper that compares only
against recurrent cores has left the obvious question unanswered.

The comparison worth making is not who retains more -- attention retains
everything by construction -- but at what price. Attention's state grows
linearly with the episode and its compute quadratically, while a recurrent or
fast-weight core keeps a fixed-size state. If Supreme matches attention's
retention curve at constant state size, that is the finding; if it does not,
that is worth knowing too.

``max_context`` turns the block into a sliding window, which is the honest way
to compare against a bounded state at long horizons.
"""
from __future__ import annotations

import torch

from wmcore.memory.attention import AttentionState, CausalSelfAttention
from wmcore.memory.base import MemoryBackbone
from wmcore.memory.registry import register_backbone


class AttentionBackbone(MemoryBackbone):
    """Causal self-attention over ``[z_t ; a_t]``.

    Parameters
    ----------
    d_ff:
        Feed-forward width. The default puts this backbone within ~1% of the
        LSTM baseline's 299,008 parameters, so the comparison is not a capacity
        comparison.
    max_context:
        ``None`` for true (unbounded) attention; an integer for a sliding
        window.
    """

    name = "attention"

    def __init__(self, input_dim: int, hidden_dim: int = 256, *, n_heads: int = 4,
                 d_ff: int = 50, max_context: int | None = None, dropout: float = 0.0):
        super().__init__(input_dim, hidden_dim)
        self.proj_in = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.attn = CausalSelfAttention(self.hidden_dim, n_heads, d_ff,
                                        max_context=max_context, dropout=dropout)

    def initial_state(self, batch_size: int, device: torch.device) -> AttentionState:
        return self.attn.initial_state(batch_size, device)

    def state_features(self, state: AttentionState) -> torch.Tensor:
        return state.last_out

    def detach_state(self, state: AttentionState) -> AttentionState:
        return state.detach_state()

    def forward(self, u: torch.Tensor, state: AttentionState | None = None):
        return self.attn(self.proj_in(u), state)

    def step(self, u_t: torch.Tensor, state: AttentionState):
        return self.attn.step(self.proj_in(u_t), state)

    def extra_metrics(self) -> dict[str, float]:
        return {}


register_backbone("attention", AttentionBackbone)
