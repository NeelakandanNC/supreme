"""M -- the memory model, and the single axis of variation in this study.

Anatomy
-------
``MemoryModule`` is the complete M model.  It is assembled from three parts::

    [z_t ; a_t]  ->  BACKBONE (recurrent core)  ->  h_t  ->  SHARED HEADS  ->  p(z_{t+1}), r_t, d_t
                     ^^^^^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^
                     the ONLY thing that differs        identical for both models

* The **backbone** is a :class:`MemoryBackbone`.  ``lstm`` reproduces the
  recurrent core of Ha & Schmidhuber's MDN-RNN; Supreme registers a
  nested-learning core under its own key.
* The **heads** -- a Mixture Density Network over z, plus optional reward and
  termination heads -- are defined here and shared verbatim.

Why the head is shared
----------------------
It is tempting to give each model "its own" output layer.  Don't.  If the
baseline used an MDN head and the challenger used, say, a plain Gaussian, then
every reported difference would confound *what the model remembers* with *how
it expresses uncertainty*, and no ablation in the paper could separate them.
Fixing the head means a difference in NLL is attributable to the recurrent core
by construction.

The state contract
------------------
A backbone's state is an opaque object (a tuple, a tensor, a dataclass of
fast-weight matrices -- whatever the core needs).  Three operations must work
on it, and they are all the rest of the codebase ever asks for:

``initial_state(batch, device)``  fresh state
``step(u, state)``                one timestep, for dreaming and for control
``state_features(state)``         a fixed-width vector, for the controller and
                                  for the benchmark's linear probes

That last one is what makes probing model-agnostic: a retention curve is
computed the same way whether the state is an LSTM cell or a stack of
fast-weight matrices.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


# =========================================================================
# Backbone contract -- implement this to add a memory layer
# =========================================================================
class MemoryBackbone(nn.Module, abc.ABC):
    """A recurrent core mapping ``u_1..u_L`` to features ``h_1..h_L``.

    Implementations must be causal: ``h_t`` may depend on ``u_{<=t}`` only.
    """

    #: Short identifier used in configs and in the registry.
    name: str = "backbone"

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

    @property
    def feature_dim(self) -> int:
        """Width of the vector handed to the controller and to probes."""
        return self.hidden_dim

    @abc.abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> Any:
        """Fresh state for a batch of sequences."""

    @abc.abstractmethod
    def forward(self, u: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        """Teacher-forced sequence pass.

        Parameters
        ----------
        u : (B, L, input_dim)
        state : backbone state or None (fresh)

        Returns
        -------
        h : (B, L, feature_dim), final_state
        """

    @abc.abstractmethod
    def step(self, u: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        """Single timestep.  ``u``: (B, input_dim) -> ``h``: (B, feature_dim)."""

    @abc.abstractmethod
    def state_features(self, state: Any) -> torch.Tensor:
        """(B, feature_dim) summary of ``state``, used by C and by probes."""

    def detach_state(self, state: Any) -> Any:
        """Detach the state from the graph (truncated BPTT across windows)."""
        return _detach_tree(state)

    def pre_optimizer_step(self, global_step: int) -> None:
        """Hook called after ``backward()`` and before ``optimizer.step()``.

        Exists for memory layers whose *own* definition includes an update
        schedule -- the Continuum Memory System updates its levels at different
        frequencies (Behrouz et al. 2025, Eq. 71), which is a property of the
        architecture, not a training trick.

        Default is a no-op, so the training loop is byte-identical for every
        backbone: the baseline neither knows nor cares that this hook exists.
        Implementations must not change *what* the optimiser is, only which of
        their own gradients are live on a given step.
        """
        return None

    def extra_metrics(self) -> dict[str, float]:
        """Optional backbone-specific diagnostics logged each epoch.

        Used by memory layers with internal dynamics worth watching (effective
        learning rates, retention gates, per-level update counts).  The baseline
        LSTM returns nothing.
        """
        return {}


def _detach_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(_detach_tree(o) for o in obj)
    if isinstance(obj, list):
        return [_detach_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _detach_tree(v) for k, v in obj.items()}
    if hasattr(obj, "detach_state"):
        return obj.detach_state()
    return obj


# =========================================================================
# Shared output heads
# =========================================================================
@dataclass
class MemoryOutput:
    """Everything M predicts at every timestep."""

    logpi: torch.Tensor      # (B, L, D, K) log mixture weights
    mu: torch.Tensor         # (B, L, D, K)
    logsigma: torch.Tensor   # (B, L, D, K)
    features: torch.Tensor   # (B, L, F) the h_t handed to the controller
    reward: torch.Tensor | None = None   # (B, L)
    done_logit: torch.Tensor | None = None  # (B, L)
    state: Any = None


class MDNHead(nn.Module):
    """Mixture Density Network over the next latent.

    Diagonal: one independent K-component 1-D mixture per latent dimension,
    exactly as in Ha & Schmidhuber (2018) Section 2.2 and in SketchRNN.  Output
    width is ``3 * K * D``.
    """

    def __init__(self, feature_dim: int, latent_dim: int, n_mixtures: int = 5):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_mixtures = int(n_mixtures)
        self.proj = nn.Linear(feature_dim, 3 * self.latent_dim * self.n_mixtures)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = h.shape
        D, K = self.latent_dim, self.n_mixtures
        out = self.proj(h).view(B, L, D, 3, K)
        logpi = F.log_softmax(out[..., 0, :], dim=-1)
        mu = out[..., 1, :]
        # Clamped for the same reason as in the VAE: the memory environments
        # contain deterministic frames whose conditional variance would
        # otherwise be driven to zero and blow up the NLL.
        logsigma = out[..., 2, :].clamp(-7.0, 7.0)
        return logpi, mu, logsigma


def mdn_nll(logpi: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    """Per-step negative log-likelihood, summed over latent dimensions.

    Returns (B, L) in nats.  Summing over D (rather than averaging) keeps the
    number comparable with the per-frame ELBO of V and is the convention in the
    World Models code release.
    """
    z = target.unsqueeze(-1)                                # (B, L, D, 1)
    log_prob = -0.5 * ((z - mu) * torch.exp(-logsigma)) ** 2 - logsigma - 0.5 * LOG_2PI
    log_mix = torch.logsumexp(logpi + log_prob, dim=-1)     # (B, L, D)
    return -log_mix.sum(dim=-1)


def mdn_sample(logpi: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor,
               temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample z ~ p(z) from the mixture, with the paper's temperature scheme.

    Temperature tau flattens the mixture weights (``logpi / tau``) *and* scales
    the component standard deviations by ``sqrt(tau)``; Ha & Schmidhuber show
    that tau > 1 is what stops a controller from exploiting an over-confident
    dream (their Section 5, "Cheating the World Model").
    """
    tau = max(1e-6, float(temperature))
    pi = F.softmax(logpi / tau, dim=-1)                     # (B, L, D, K)
    flat = pi.reshape(-1, pi.shape[-1])
    idx = torch.multinomial(flat, 1, generator=generator).view(*pi.shape[:-1], 1)
    mu_sel = torch.gather(mu, -1, idx).squeeze(-1)
    sigma_sel = torch.gather(logsigma, -1, idx).squeeze(-1).exp()
    noise = torch.randn(mu_sel.shape, device=mu_sel.device, dtype=mu_sel.dtype,
                        generator=generator)
    return mu_sel + sigma_sel * noise * math.sqrt(tau)


def mdn_mean(logpi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Mixture mean E[z] -- the deterministic prediction used for MSE metrics."""
    return (logpi.exp() * mu).sum(dim=-1)


# =========================================================================
# The full M model
# =========================================================================
class MemoryModule(nn.Module):
    """Backbone + shared heads.  Constructed by :func:`wmcore.memory.build_memory`."""

    def __init__(
        self,
        backbone: MemoryBackbone,
        latent_dim: int,
        action_dim: int,
        *,
        n_mixtures: int = 5,
        predict_reward: bool = True,
        predict_done: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        expected = latent_dim + action_dim
        if backbone.input_dim != expected:
            raise ValueError(
                f"backbone.input_dim={backbone.input_dim} but M is fed "
                f"[z ({latent_dim}) ; a ({action_dim})] = {expected}"
            )
        self.backbone = backbone
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.temperature = float(temperature)

        self.head = MDNHead(backbone.feature_dim, latent_dim, n_mixtures)
        self.reward_head = nn.Linear(backbone.feature_dim, 1) if predict_reward else None
        self.done_head = nn.Linear(backbone.feature_dim, 1) if predict_done else None

    # ------------------------------------------------------------ shapes --
    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    @property
    def controller_input_dim(self) -> int:
        """Width of [z_t ; h_t], the controller's observation."""
        return self.latent_dim + self.feature_dim

    # ------------------------------------------------------------ passes --
    def initial_state(self, batch_size: int, device: torch.device) -> Any:
        return self.backbone.initial_state(batch_size, device)

    def forward(self, z: torch.Tensor, action: torch.Tensor, state: Any = None) -> MemoryOutput:
        """Teacher-forced pass.  ``z``: (B, L, D), ``action``: (B, L, A)."""
        u = torch.cat([z, action], dim=-1)
        h, new_state = self.backbone(u, state)
        logpi, mu, logsigma = self.head(h)
        return MemoryOutput(
            logpi=logpi, mu=mu, logsigma=logsigma, features=h,
            reward=self.reward_head(h).squeeze(-1) if self.reward_head is not None else None,
            done_logit=self.done_head(h).squeeze(-1) if self.done_head is not None else None,
            state=new_state,
        )

    def step(self, z: torch.Tensor, action: torch.Tensor, state: Any) -> MemoryOutput:
        """One timestep.  ``z``: (B, D), ``action``: (B, A).  Used for dreaming."""
        u = torch.cat([z, action], dim=-1)
        h, new_state = self.backbone.step(u, state)
        h_seq = h.unsqueeze(1)
        logpi, mu, logsigma = self.head(h_seq)
        return MemoryOutput(
            logpi=logpi, mu=mu, logsigma=logsigma, features=h_seq,
            reward=self.reward_head(h_seq).squeeze(-1) if self.reward_head is not None else None,
            done_logit=self.done_head(h_seq).squeeze(-1) if self.done_head is not None else None,
            state=new_state,
        )

    def state_features(self, state: Any) -> torch.Tensor:
        return self.backbone.state_features(state)

    def sample_next(self, out: MemoryOutput, temperature: float | None = None) -> torch.Tensor:
        tau = self.temperature if temperature is None else temperature
        return mdn_sample(out.logpi, out.mu, out.logsigma, tau)

    # -------------------------------------------------------------- loss --
    def loss(
        self,
        out: MemoryOutput,
        batch: dict[str, torch.Tensor],
        *,
        reward_weight: float = 1.0,
        done_weight: float = 1.0,
        critical_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Masked training objective, identical for every backbone.

        Also returns ``nll_memory_critical``: the same NLL restricted to the
        transitions the environment flagged as requiring memory.  On the
        synthetic suite that single number is the cleanest read-out of what this
        study is about -- the average NLL is dominated by easy, locally
        predictable frames and moves very little between models.
        """
        mask = batch["mask"]                                  # (B, L)
        denom = mask.sum().clamp(min=1.0)

        nll = mdn_nll(out.logpi, out.mu, out.logsigma, batch["z_next"])  # (B, L)

        crit = batch.get("memory_critical")
        if critical_weight != 1.0 and crit is not None:
            # Upweight the transitions that cannot be predicted without memory.
            # See MemoryConfig.critical_loss_weight for why this is necessary and
            # why it does not compromise the comparison: the weight is a property
            # of the shared objective, applied to every backbone alike.
            w = mask * (1.0 + (critical_weight - 1.0) * crit)
            loss_z = (nll * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss_z = (nll * mask).sum() / denom

        # Reported NLL is always the *unweighted* per-step average, so numbers
        # remain comparable across different values of critical_loss_weight.
        losses = {"loss": loss_z,
                  "nll_z": ((nll * mask).sum() / denom).detach()}

        if crit is not None:
            crit_mask = crit * mask
            crit_denom = crit_mask.sum()
            # NaN, not zero, when a batch happens to contain no critical step:
            # the aggregator skips NaN, whereas a zero would be averaged in and
            # would drag the reported value below its true level.
            losses["nll_memory_critical"] = (
                (nll * crit_mask).sum() / crit_denom if crit_denom > 0
                else torch.full((), float("nan"), device=nll.device)
            ).detach()

        with torch.no_grad():
            pred = mdn_mean(out.logpi, out.mu)
            mse = ((pred - batch["z_next"]) ** 2).sum(-1)
            losses["mse_z"] = ((mse * mask).sum() / denom).detach()

        if out.reward is not None and "reward" in batch:
            r = ((out.reward - batch["reward"]) ** 2 * mask).sum() / denom
            losses["loss"] = losses["loss"] + reward_weight * r
            losses["mse_reward"] = r.detach()

        if out.done_logit is not None and "done" in batch:
            d = (F.binary_cross_entropy_with_logits(
                out.done_logit, batch["done"], reduction="none") * mask).sum() / denom
            losses["loss"] = losses["loss"] + done_weight * d
            losses["bce_done"] = d.detach()

        return losses
