"""The baseline memory core: the recurrent half of the MDN-RNN.

Reference
---------
D. Ha and J. Schmidhuber, *World Models*, 2018 (arXiv:1803.10122), Section 2.2.
Their M model is an LSTM whose output feeds a Mixture Density Network that
parameterises ``P(z_{t+1} | a_t, z_t, h_t)``.  In this repository the MDN half
lives in :mod:`wmcore.memory.base` and is shared with every other backbone; what
is defined here is precisely the part that Supreme replaces -- the recurrent
core.

Sizing follows the paper: a single LSTM layer with 256 hidden units, taking
``[z_t ; a_t]`` as input.  With a 32-d latent, a 3-d CarRacing action and 5
mixture components the full M model comes to ~422 k parameters, matching the
count in their Appendix.

Nothing clever happens in this file, and that is the point: it is the control
condition.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from wmcore.memory.base import MemoryBackbone
from wmcore.memory.registry import register_backbone


class LSTMBackbone(MemoryBackbone):
    """Single- or multi-layer LSTM recurrent core.

    Parameters
    ----------
    input_dim:
        ``latent_dim + action_dim``.  Fixed by the framework.
    hidden_dim:
        Recurrent width.  256 in the paper.  Held equal across models so the
        controller's input size -- and therefore C's parameter count and the
        CMA-ES search dimension -- is identical in both legs of the study.
    num_layers:
        Stacked LSTM layers.  1 in the paper.
    dropout:
        Applied between layers when ``num_layers > 1``.  Off by default; the
        paper uses none and adding it to only one leg would be a confound.
    """

    name = "lstm"

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__(input_dim, hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    # ------------------------------------------------------------- state --
    def initial_state(self, batch_size: int, device: torch.device):
        shape = (self.num_layers, batch_size, self.hidden_dim)
        zeros = torch.zeros(shape, device=device)
        return (zeros, zeros.clone())

    def state_features(self, state) -> torch.Tensor:
        """Top-layer hidden state h_t.

        This is exactly what Ha & Schmidhuber feed to the controller alongside
        z_t (their Section 2.3): C sees ``[z_t ; h_t]``.
        """
        h, _ = state
        return h[-1]

    # ------------------------------------------------------------ passes --
    def forward(self, u: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple]:
        if state is None:
            state = self.initial_state(u.shape[0], u.device)
        h_seq, new_state = self.rnn(u, state)
        return h_seq, new_state

    def step(self, u: torch.Tensor, state) -> tuple[torch.Tensor, tuple]:
        h_seq, new_state = self.rnn(u.unsqueeze(1), state)
        return h_seq.squeeze(1), new_state


class GRUBackbone(MemoryBackbone):
    """GRU core -- a cheap third data point, not part of the headline claim.

    Included because a reviewer will ask whether any difference between the
    baseline and Supreme is just "LSTM is a weak recurrent net".  A GRU run
    costs a few minutes and answers that.
    """

    name = "gru"

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__(input_dim, hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True)

    def initial_state(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def state_features(self, state) -> torch.Tensor:
        return state[-1]

    def forward(self, u: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(u.shape[0], u.device)
        return self.rnn(u, state)

    def step(self, u: torch.Tensor, state):
        h_seq, new_state = self.rnn(u.unsqueeze(1), state)
        return h_seq.squeeze(1), new_state


register_backbone("lstm", LSTMBackbone)
register_backbone("gru", GRUBackbone)
