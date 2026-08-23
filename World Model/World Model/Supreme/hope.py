"""Hope -- the Supreme memory core.

Implements Section 8.3 of Behrouz et al. (NeurIPS 2025), Equations 94-97: a
self-modifying Titans block whose output is passed through a Continuum Memory
System chain.

    o_t = M_mem,t-1(q_t)                                (Titans, Eqs. 94-96)
    y_t = MLP^(f_k)( ... MLP^(f_1)(o_t) )               (CMS,    Eq. 97)

The paper's rationale (Section 8.3): CMS has large capacity but a simple
learning rule, while self-modifying Titans has small capacity but a very
expressive one, so the two are complementary. That claim is exactly the kind of
thing an ablation should check rather than assume, which is why this file
registers three backbones rather than one:

===================  ==========================================================
``supreme``          Titans + CMS. Hope as published; the headline model.
``supreme-titans``   Titans alone. Isolates the self-modifying memory.
``supreme-cms``      LSTM + CMS. Isolates CMS by bolting it onto the *baseline*
                     recurrent core, so any gain is attributable to the
                     multi-frequency knowledge store and nothing else.
===================  ==========================================================

Together with ``lstm`` that is a four-column table from one implementation, and
it can distinguish "Hope helps" from "one of Hope's two halves helps".

Everything outside this file is untouched: the environments, the data, V, the
shared MDN head, the loss, the optimiser, C and the whole benchmark are the same
objects the baseline uses. See ``docs/design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .cms import ContinuumMemorySystem
from .titans import SelfModifyingTitans, TitansState
from wmcore.memory.attention import CausalSelfAttention
from wmcore.memory.base import MemoryBackbone
from wmcore.memory.registry import register_backbone


@dataclass
class HopeState:
    """Mixer state plus the last emitted feature vector.

    ``last_out`` is what the controller and the probes read.  It is the feature
    *before* the current observation has been consumed, matching the LSTM
    baseline's ``h_t`` convention exactly -- see ``wmcore/controller/agent.py``.
    """

    mixer: Any
    last_out: torch.Tensor

    def detach_state(self) -> "HopeState":
        mixer = (self.mixer.detach_state() if hasattr(self.mixer, "detach_state")
                 else tuple(m.detach() for m in self.mixer))
        return HopeState(mixer=mixer, last_out=self.last_out.detach())


class HopeBackbone(MemoryBackbone):
    """Sequence mixer + Continuum Memory System.

    Parameters
    ----------
    input_dim, hidden_dim:
        Fixed by the framework.  ``hidden_dim`` is also ``d_model`` and
        ``feature_dim``, so the controller sees the same width as with the
        baseline and the CMA-ES search dimension is unchanged.
    mixer:
        ``"titans"`` (self-modifying Titans) or ``"lstm"`` (the baseline core,
        used for the CMS-only ablation).
    cms_periods:
        Update periods ``C^(l)`` per CMS level, fastest first.  An empty tuple
        disables CMS entirely.
    n_heads, chunk_size, eta_max, alpha_min:
        Passed to :class:`~.titans.SelfModifyingTitans`.
    cms_ff:
        Hidden width inside each CMS block.  This and ``n_heads`` are the two
        dials for matching the baseline's ~299 k backbone parameters; the
        constructor logs the count and ``wmcore.bench.cost`` reports it.
    """

    name = "supreme"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        *,
        mixer: str = "titans",
        n_heads: int = 4,
        chunk_size: int = 8,
        conv_window: int = 4,
        eta_max: float = 0.5,
        alpha_min: float = 0.0,
        cms_periods: tuple[int, ...] = (1, 4, 16),
        cms_ff: int = 48,
        attn_ff: int = 50,
        max_context: int | None = None,
        cms_aggregate: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim, hidden_dim)
        self.mixer_kind = mixer

        if mixer == "titans":
            self.proj_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.mixer = SelfModifyingTitans(
                d_model=self.hidden_dim, n_heads=n_heads, chunk_size=chunk_size,
                conv_window=conv_window, eta_max=eta_max, alpha_min=alpha_min,
            )
        elif mixer == "attention":
            self.proj_in = nn.Identity()
            self.mixer = CausalSelfAttention(self.hidden_dim, n_heads, attn_ff,
                                             max_context=max_context)
            self.pre = nn.Linear(self.input_dim, self.hidden_dim)
        elif mixer == "lstm":
            # No input projection: the LSTM consumes [z ; a] directly, so this
            # variant is *literally* the baseline core plus CMS and the ablation
            # differs from the baseline in exactly one component.
            self.proj_in = nn.Identity()
            self.mixer = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        else:
            raise ValueError(
                f"unknown mixer {mixer!r}; use 'titans', 'attention' or 'lstm'")

        self.cms = (ContinuumMemorySystem(self.hidden_dim, cms_ff, tuple(cms_periods),
                                          aggregate=cms_aggregate, dropout=dropout)
                    if cms_periods else None)

    # ------------------------------------------------------------- state --
    def initial_state(self, batch_size: int, device: torch.device) -> HopeState:
        if self.mixer_kind in ("titans", "attention"):
            mixer = self.mixer.initial_state(batch_size, device)
        else:
            zeros = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            mixer = (zeros, zeros.clone())
        return HopeState(mixer=mixer,
                         last_out=torch.zeros(batch_size, self.hidden_dim, device=device))

    def state_features(self, state: HopeState) -> torch.Tensor:
        return state.last_out

    def detach_state(self, state: HopeState) -> HopeState:
        return state.detach_state()

    # ------------------------------------------------------------ passes --
    def _post(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the CMS chain (Eq. 97) to the mixer output."""
        return self.cms(h) if self.cms is not None else h

    def _embed(self, u: torch.Tensor) -> torch.Tensor:
        return self.pre(u) if self.mixer_kind == "attention" else self.proj_in(u)

    def forward(self, u: torch.Tensor, state: HopeState | None = None
                ) -> tuple[torch.Tensor, HopeState]:
        if state is None:
            state = self.initial_state(u.shape[0], u.device)
        h, mixer_state = self.mixer(self._embed(u), state.mixer)
        y = self._post(h)
        return y, HopeState(mixer=mixer_state, last_out=y[:, -1])

    def step(self, u_t: torch.Tensor, state: HopeState) -> tuple[torch.Tensor, HopeState]:
        x = self._embed(u_t)
        if self.mixer_kind == "lstm":
            h_seq, mixer_state = self.mixer(x.unsqueeze(1), state.mixer)
            h = h_seq.squeeze(1)
        else:
            h, mixer_state = self.mixer.step(x, state.mixer)
        y = self._post(h.unsqueeze(1)).squeeze(1)
        return y, HopeState(mixer=mixer_state, last_out=y)

    # -------------------------------------------------------------- hooks --
    def pre_optimizer_step(self, global_step: int) -> None:
        """Forward Equation 71's update schedule to the CMS chain."""
        if self.cms is not None:
            self.cms.pre_optimizer_step(global_step)

    def extra_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.mixer_kind == "titans":
            out.update(self.mixer.metrics())
        if self.cms is not None:
            out.update(self.cms.metrics())
        return out


# --------------------------------------------------------------------------
# Registration.
#
# Parameter budgets (input_dim 34, hidden_dim 256), against the baseline LSTM's
# 299,008:
#
#   supreme          ~300.7 k   matched to the baseline within 1%
#   supreme-titans   ~224.5 k   strictly smaller: it is `supreme` with CMS removed
#   supreme-cms      ~375.2 k   strictly larger: it is the BASELINE plus CMS
#
# Only `supreme` is parameter-matched, and only `supreme` needs to be: it is the
# model making the claim. The two ablations are subsets/supersets by
# construction -- `supreme-cms` cannot be matched to the baseline because it *is*
# the baseline plus extra blocks -- so their counts are reported rather than
# equalised, and each is read against the model it extends or reduces.
# `cms_ff` and `n_heads` are the dials if you need to re-match after changing
# hidden_dim; `wmcore.bench.cost` reports the counts on every run.
# --------------------------------------------------------------------------
def _hope(**kwargs) -> HopeBackbone:
    return HopeBackbone(mixer="titans", **kwargs)


def _titans_only(**kwargs) -> HopeBackbone:
    kwargs.setdefault("cms_periods", ())
    return HopeBackbone(mixer="titans", **kwargs)


def _cms_only(**kwargs) -> HopeBackbone:
    return HopeBackbone(mixer="lstm", **kwargs)


def _hope_attention(**kwargs) -> HopeBackbone:
    """The paper's own Hope-Attention variant (Section 8.3): softmax attention
    in place of the self-modifying Titans block, still followed by CMS."""
    return HopeBackbone(mixer="attention", **kwargs)


register_backbone("supreme", _hope)
register_backbone("supreme-titans", _titans_only)
register_backbone("supreme-cms", _cms_only)
register_backbone("hope-attention", _hope_attention)
