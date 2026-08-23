"""Self-modifying (self-referential) Titans.

Implements Section 8.1-8.2 of Behrouz et al., *Nested Learning: The Illusion of
Deep Learning Architecture* (NeurIPS 2025), Equations 86-93.

The idea
--------
An ordinary attention block projects the input to keys, values and queries with
weights that are frozen after pre-training.  Self-modifying Titans replaces
those projections with **associative memories that keep learning during the
sequence**, and -- the "self-modifying" part -- lets each memory generate its
own regression targets rather than reading them from the data::

    q_t   = x_t W_q                                    (the only fixed projection)
    k_t   = M_k,t-1(x_t)      v_t = M_v,t-1(x_t)
    eta_t = M_eta,t-1(x_t)    alpha_t = M_alpha,t-1(x_t)
    v^_*,t = M_*,t-1(v_t)                              (Eq. 87: own values)
    M_*,t = M_*,t-1 (alpha_t I - eta_t k_t k_t^T)
            - eta_t grad L(M_*,t-1 ; k_t, v^_*,t)      (Eq. 88)
    o_t   = M_mem,t-1(q_t)

with ``L`` the L2 regression loss and every memory a residual block
``M(.) = (.) + W1 sigma(W2 .)`` (Eq. 89).  ``eta_t`` is a data-dependent inner
learning rate and ``alpha_t`` a data-dependent retention (forget) gate; the
initial states ``M_*,0`` are ordinary trainable parameters, meta-learned across
sequences.

Choices made here, and why
--------------------------
**Linear (matrix-valued) memories by default.**  Section 8.2 derives the
closed-form recurrence for exactly this case (Eq. 93), and it is the form that
admits an exact update without calling autograd inside the recurrence.  The
residual structure of Eq. 89 is kept -- retrieval is ``x + M x``, not ``M x`` --
so the memory learns a correction to identity, which is what makes the initial
state a sensible starting point.

**The chunk is semantic, not just an optimisation.**  Section 8.2 computes
``k, v, eta, alpha``, the generated values and the gradients from the memory
state at the *start* of the chunk.  That changes what the model computes, not
only how fast it computes it, so it is implemented faithfully and exposed as
``chunk_size``.  ``chunk_size=1`` recovers the fully-online form of Eqs. 86-88.

**One code path for `forward` and `step`.**  The paper's parallel-within-chunk
dual form is a speed optimisation; at the sequence lengths in this study
(L <= 69) it buys nothing, and a separate fast path is the classic way to end up
with a model whose training-time and rollout-time behaviour silently differ.
Both entry points call the same ``_advance``, so
``tests/test_core.py::test_backbone_forward_matches_step`` holds by construction.

**Ambiguity in the paper, resolved explicitly.**  Equation 94 lists a memory for
``q`` while the surrounding text states that ``q_t = x_t W_q`` is "the only
non-adaptive projection".  We follow the text: memories are
``{k, v, eta, alpha, mem}`` and ``q`` uses a fixed learned projection.
Similarly Eq. 90 indexes the retrieval memory by ``C*ceil(t/C)``, which is not
causal for ``t`` inside a chunk; we use the chunk *start* (floor), which is.

**One stability measure the linear form requires, and why.**  Written exactly
as Eq. 88, this recurrence diverges: measured here, the memory norm reaches
1e10 by step 80 and overflows by step 96.  The mechanism is specific and worth
stating, because it is a property of self-modification rather than a bug.
Expanding the update,

    M_t = M_{t-1}(alpha_t I - eta_t k k^T) - eta_t (M_ref k - v^) k^T

the generated target ``v^ = M_ref(v)`` is itself a function of the memory, and
``v`` is in turn ``M_ref(x)``.  So the write term contains ``+eta M_ref v k^T``,
which is *quadratic* in ||M||: a memory that generates its own values feeds its
own magnitude back into itself.  In the paper the memories are 2-layer MLPs
sitting inside a normalised stack, which absorbs this; the linear form of Eq. 93
has nothing to absorb it.

The fix is a **projection onto a Frobenius ball** after each update
(``mem_max_norm``, see :meth:`_project`).  The update rule itself is left
*exactly* as Eq. 88 -- no normalisation of the target, no constraint tying the
gates -- and the projection is inactive below the radius.  At the study's
operating point (episodes of 37 steps) it never engages: the memory norm settles
around 2.5 against a radius of 20.  It engages only on the long rollouts where
the divergence would otherwise appear, and ``titans_clip_rate`` in the epoch
metrics reports exactly how often, so a run where the model is being constrained
rather than guarded is visible rather than silent.

Two less invasive-looking alternatives were tried first and rejected; see
:meth:`_project` for what each cost.

``tests/test_core.py::test_titans_memory_stays_finite_over_a_long_sequence``
pins this at 300 steps, longer than any config in the study uses, because the
dream environment rolls 300 steps and CarRacing episodes are 250.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The memories that are updated in-context.  ``mem`` is the one that answers
#: queries; the other four decide how the input is encoded and how fast the
#: memories move -- that is the "self-referential" part.
SLOTS = ("k", "v", "eta", "alpha", "mem")


@dataclass
class TitansState:
    """Everything the recurrence needs to continue from step ``t``.

    ``M`` is the live memory; ``M_ref`` is its snapshot at the start of the
    current chunk, which is what generates keys/values/gates and what the
    inner gradient is taken at (Eq. 90).

    Both are single stacked tensors of shape ``(B, 5, H, D, D)`` -- one slice per
    entry of :data:`SLOTS`, in that order -- rather than five separate matrices.
    That is purely a layout choice: it lets the five memories be read and updated
    with one batched ``einsum`` each instead of five, which on MPS is the
    difference between a usable training run and an unusable one (kernel-launch
    overhead dominates at these tensor sizes). ``_advance_reference`` keeps the
    unstacked, equation-by-equation version, and a test asserts the two agree.
    """

    M: torch.Tensor                   # (B, S, H, D, D)
    M_ref: torch.Tensor               # (B, S, H, D, D)
    conv_buf: torch.Tensor            # (B, kernel-1, d_model) causal conv history
    last_out: torch.Tensor            # (B, d_model) output of step t-1
    t: int                            # steps consumed, for chunk boundaries

    def detach_state(self) -> "TitansState":
        return TitansState(
            M=self.M.detach(), M_ref=self.M_ref.detach(),
            conv_buf=self.conv_buf.detach(), last_out=self.last_out.detach(),
            t=self.t,
        )


class SelfModifyingTitans(nn.Module):
    """The self-referential associative-memory block of Eqs. 86-93.

    Parameters
    ----------
    d_model:
        Working width.  Equals the backbone's ``hidden_dim`` so the controller's
        input size matches the baseline exactly.
    n_heads:
        Memories are block-diagonal across heads: ``n_heads`` independent
        ``d_head x d_head`` matrices rather than one ``d_model x d_model``.  This
        is what keeps the parameter count near the LSTM's ~299 k; a single full
        matrix per slot would be 5 x 65 k of state per sequence.
    chunk_size:
        Steps between refreshes of the reference memory (Section 8.2).
    eta_max:
        Ceiling on the inner learning rate.  Unbounded ``eta`` makes the
        recurrence diverge within a few hundred steps; the paper's experiments
        are at much larger scale where the surrounding normalisation absorbs it.
    alpha_min:
        Floor on the retention gate, so a single confident step cannot erase the
        whole memory.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        chunk_size: int = 8,
        conv_window: int = 4,
        eta_max: float = 0.5,
        alpha_min: float = 0.0,
        mem_max_norm: float = 20.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.chunk_size = max(1, int(chunk_size))
        self.conv_window = max(1, int(conv_window))
        self.eta_max = float(eta_max)
        self.alpha_min = float(alpha_min)
        self.mem_max_norm = float(mem_max_norm)

        # Local causal convolution, window 4 (Section 8.3).  Depthwise: it mixes
        # over time only, leaving cross-feature mixing to the memories.
        self.conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=self.conv_window,
                              groups=self.d_model, bias=False)

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

        # Meta-learned initial memory states, M_*,0.  Small init: the memories
        # are residual corrections to identity, so starting near zero starts the
        # block near a plain projection and lets it earn its deviations.
        for slot in SLOTS:
            self.register_parameter(
                f"M0_{slot}",
                nn.Parameter(torch.randn(self.n_heads, self.d_head, self.d_head)
                             * (0.02 / math.sqrt(self.d_head))),
            )

        # Read-outs that turn the eta/alpha memories' vector output into the two
        # per-head scalars the update rule needs.
        self.eta_read = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.eta_bias = nn.Parameter(torch.full((self.n_heads,), -1.0))
        self.alpha_read = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
        self.alpha_bias = nn.Parameter(torch.full((self.n_heads,), 2.0))  # start near "retain"

        self._last_eta = 0.0
        self._last_alpha = 0.0
        self._clip_rate = 0.0

    # ------------------------------------------------------------- state --
    def initial_state(self, batch_size: int, device: torch.device) -> TitansState:
        M0 = torch.stack([getattr(self, f"M0_{slot}") for slot in SLOTS], dim=0)
        M = M0.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()
        return TitansState(
            M=M,
            M_ref=M.clone(),
            conv_buf=torch.zeros(batch_size, self.conv_window - 1, self.d_model,
                                 device=device),
            last_out=torch.zeros(batch_size, self.d_model, device=device),
            t=0,
        )

    # ------------------------------------------------------------ pieces --
    @staticmethod
    def _retrieve(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Residual associative read, Eq. 89 with a linear memory.

        ``M``: (B, H, D, D), ``x``: (B, H, D) -> (B, H, D).
        """
        return x + torch.einsum("bhij,bhj->bhi", M, x)

    def _project(self, M: torch.Tensor) -> torch.Tensor:
        """Project each head's memory onto a Frobenius ball of radius ``mem_max_norm``.

        This is the whole stability mechanism, and it is deliberately the least
        invasive one available: the update rule is *exactly* Equation 88, and the
        projection is inactive whenever the norm is below the radius -- which,
        with the radius set from the measured operating range, is essentially
        always during normal operation.  ``clip_rate`` in ``metrics()`` reports
        how often it actually engages; if that is not near zero the radius is too
        tight and the model is being constrained rather than merely guarded.

        Two alternatives were tried first and are recorded here because both are
        tempting and both are worse:

        * normalising the generated target ``v^`` -- stabilises the recurrence,
          but destroys the magnitude of the very quantity the update regresses
          on.  Measured: decision accuracy at corridor 32 fell from 1.000 to
          chance (0.48) while the training curve looked healthy for the first
          ~25 epochs, so the damage was invisible early.
        * tying retention to the write rate (``alpha <= 1 - eta``) -- harmless,
          but pointless: the learned ``alpha`` settles near 0.77 either way, so
          the constraint never binds and it buys no stability on its own.
        """
        norm = M.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
        scale = (self.mem_max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
        self._clip_rate = float((scale < 1.0).float().mean().detach())
        return M * scale

    @staticmethod
    def _retrieve_stacked(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The same read for every slot at once.

        ``M``: (B, S, H, D, D), ``x``: (B, H, D) -> (B, S, H, D).
        """
        return x.unsqueeze(1) + torch.einsum("bshij,bhj->bshi", M, x)

    def _local_conv(self, x_t: torch.Tensor, buf: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise conv over the last ``conv_window`` inputs."""
        if self.conv_window == 1:
            return x_t, buf
        window = torch.cat([buf, x_t.unsqueeze(1)], dim=1)      # (B, W, d)
        out = self.conv(window.transpose(1, 2)).squeeze(-1)     # (B, d)
        return out, window[:, 1:]

    # ----------------------------------------------------------- advance --
    def _prepare(self, u_t: torch.Tensor, state: TitansState):
        """Shared front half of a step: chunk refresh, norm, conv, q/k/v/eta/alpha.

        Both :meth:`_advance` and :meth:`_advance_reference` call this, so the
        two differ only in how the memory update is arranged -- which is what the
        equivalence test is actually testing.
        """
        B = u_t.shape[0]
        H, D = self.n_heads, self.d_head

        # Refresh the chunk reference before the step that opens a new chunk.
        M_ref = state.M if state.t % self.chunk_size == 0 else state.M_ref

        x = self.norm_in(u_t)
        x, conv_buf = self._local_conv(x, state.conv_buf)
        xh = x.view(B, H, D)

        # --- generate keys, values, gates from the chunk-start memories ------
        # Slots 0..3 are k, v, eta, alpha and all read the same input, so one
        # batched retrieval covers them.
        kvea = self._retrieve_stacked(M_ref[:, :4], xh)          # (B, 4, H, D)
        k, v, e, a = kvea[:, 0], kvea[:, 1], kvea[:, 2], kvea[:, 3]

        # L2-normalised q and k (Section 8.3).  Normalising k also bounds
        # ||k k^T||, which is what keeps the decay term in Eq. 88 stable.
        q = F.normalize(self.W_q(x).view(B, H, D), dim=-1)
        k = F.normalize(k, dim=-1)

        eta = self.eta_max * torch.sigmoid(
            (e * self.eta_read).sum(-1) + self.eta_bias)                  # (B, H)
        alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(
            (a * self.alpha_read).sum(-1) + self.alpha_bias)              # (B, H)

        # --- the block's output, read from the memory BEFORE its update -----
        o = self._retrieve(M_ref[:, SLOTS.index("mem")], q).reshape(B, self.d_model)
        y = self.norm_out(self.W_out(o))

        self._last_eta = float(eta.detach().mean())
        self._last_alpha = float(alpha.detach().mean())
        return y, M_ref, conv_buf, k, v, eta, alpha

    def _advance(self, u_t: torch.Tensor, state: TitansState
                 ) -> tuple[torch.Tensor, TitansState]:
        """One timestep of Eqs. 86-88 under the chunked schedule of Eq. 90.

        All five memories are updated with a single batched einsum.  Verified
        against :meth:`_advance_reference` by
        ``tests/test_core.py::test_titans_fast_path_matches_reference``.
        """
        D = self.d_head
        y, M_ref, conv_buf, k, v, eta, alpha = self._prepare(u_t, state)

        eta_e = eta[:, None, :, None, None]        # (B, 1, H, 1, 1)
        alpha_e = alpha[:, None, :, None, None]
        kk = (k.unsqueeze(-1) * k.unsqueeze(-2)).unsqueeze(1)   # (B, 1, H, D, D)
        eye = torch.eye(D, device=k.device).view(1, 1, 1, D, D)
        decay = alpha_e * eye - eta_e * kk                       # (B, 1, H, D, D)

        # Eq. 87: each memory generates its own target from the value stream.
        v_hat = self._retrieve_stacked(M_ref, v)                 # (B, S, H, D)
        # grad of 1/2 ||M k - v_hat||^2 at the chunk-start state (Eq. 90).
        resid = torch.einsum("bshij,bhj->bshi", M_ref, k) - v_hat
        grad = resid.unsqueeze(-1) * k[:, None, :, None, :]      # (B, S, H, D, D)
        new_M = self._project(
            torch.einsum("bshij,bzhjl->bshil", state.M, decay) - eta_e * grad)

        return y, TitansState(M=new_M, M_ref=M_ref, conv_buf=conv_buf,
                              last_out=y, t=state.t + 1)

    def _advance_reference(self, u_t: torch.Tensor, state: TitansState
                           ) -> tuple[torch.Tensor, TitansState]:
        """Literal, slot-by-slot transcription of Eqs. 86-88.

        Slower and never used in training.  It exists so the equations in the
        paper can be read straight off the code, and so the fast path has
        something to be checked against -- a hand-optimised recurrence that
        nothing verifies is how a subtly wrong model reaches a submission.
        """
        D = self.d_head
        y, M_ref, conv_buf, k, v, eta, alpha = self._prepare(u_t, state)

        eta_e = eta[..., None, None]
        alpha_e = alpha[..., None, None]
        kk = k.unsqueeze(-1) * k.unsqueeze(-2)                   # (B, H, D, D)
        decay = alpha_e * torch.eye(D, device=k.device).expand_as(kk) - eta_e * kk

        slices = []
        for i, _slot in enumerate(SLOTS):
            M_ref_s = M_ref[:, i]
            v_hat = self._retrieve(M_ref_s, v)
            resid = torch.einsum("bhij,bhj->bhi", M_ref_s, k) - v_hat
            grad = resid.unsqueeze(-1) * k.unsqueeze(-2)
            slices.append(
                torch.einsum("bhij,bhjl->bhil", state.M[:, i], decay) - eta_e * grad
            )
        new_M = self._project(torch.stack(slices, dim=1))

        return y, TitansState(M=new_M, M_ref=M_ref, conv_buf=conv_buf,
                              last_out=y, t=state.t + 1)

    # ------------------------------------------------------------ passes --
    def forward(self, u: torch.Tensor, state: TitansState | None = None
                ) -> tuple[torch.Tensor, TitansState]:
        """Sequence pass: the same ``_advance`` applied step by step.

        Section 8.2 describes a chunk-parallel dual form, and everything that
        reads the *chunk-start* memory -- the convolution, the key/value/gate
        generation, the self-generated targets, the inner gradients and the block
        outputs -- genuinely can be computed for a whole chunk at once, leaving
        only ``M_t = M_{t-1} decay_t - eta_t grad_t`` sequential.

        That version was implemented and measured against this one, and it is not
        used, because it is not faster here: 1.01x at L=37, 0.98x at L=69, 1.05x
        at L=250. The cost is not kernel launches, it is the backward pass
        through the sequential memory recurrence, which the chunked form does not
        remove. Given no measured gain, one code path shared with :meth:`step` is
        worth more than a second implementation that has to be kept in sync --
        the risk being a model whose training-time and rollout-time behaviour
        silently differ.

        This is also the honest cost line for the paper: Supreme's ~2 s training
        step against the LSTM's ~10 ms is inherent to a per-timestep inner
        optimisation, not an artefact of a lazy implementation, and it is the
        real reason a genuinely parallel formulation matters at scale.
        """
        if state is None:
            state = self.initial_state(u.shape[0], u.device)
        outs = []
        for t in range(u.shape[1]):
            y, state = self._advance(u[:, t], state)
            outs.append(y)
        return torch.stack(outs, dim=1), state

    def step(self, u_t: torch.Tensor, state: TitansState) -> tuple[torch.Tensor, TitansState]:
        return self._advance(u_t, state)

    def metrics(self) -> dict[str, float]:
        return {"titans_eta_mean": self._last_eta,
                "titans_alpha_mean": self._last_alpha,
                "titans_clip_rate": self._clip_rate}
