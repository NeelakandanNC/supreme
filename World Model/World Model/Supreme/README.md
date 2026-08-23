# Supreme — the nested-learning memory core

**Status: implemented.** Hope (Candidate C below) is in `hope.py`, built from
`titans.py` and `cms.py` as two independently switchable parts.

Source paper: Behrouz, Razaviyayn, Zhong, Mirrokni, *Nested Learning: The
Illusion of Deep Learning Architecture*, NeurIPS 2025 (arXiv:2512.24695).

## What is registered

| key | composition | parameters (input 34, hidden 256) |
|---|---|---|
| `supreme` | self-modifying Titans → CMS chain (Hope, §8.3, Eqs. 94–97) | 300,696 (+0.6% vs baseline) |
| `supreme-titans` | Titans alone (§8.1–8.2, Eqs. 86–93) | 224,520 (−24.9%) |
| `supreme-cms` | baseline LSTM → CMS chain (§7.1, Eqs. 70–71) | 375,184 (+25.5%) |

Only `supreme` is parameter-matched to the baseline's 299,008, and only it needs
to be — it is the model making the claim. The two ablations are a strict subset
and a strict superset of it by construction (`supreme-cms` *is* the baseline plus
CMS and so cannot be matched to it), so their counts are reported rather than
equalised and each is read against the model it extends or reduces.

```bash
python wm.py study -c configs/tmaze.yaml \
    --backbones lstm,supreme,supreme-titans,supreme-cms
```

## Two places the paper is ambiguous, and how each was resolved

Both are flagged in the source, because a reader reproducing this needs to know
where a judgement call was made.

**1. Is `q` produced by a memory?** Equation 94 lists a memory for `q` among
`{k, v, q, η, α, memory}`, while the surrounding text of §8.1 states that
`q_t = x_t W_q` is "the only non-adaptive projection". We follow the text:
memories are `{k, v, η, α, mem}` and `q` uses a fixed learned projection.

**2. What is `L` in the CMS update (Eq. 71)?** The equation is explicit that it
is "the objective of choice for the task at hand, e.g. … next token prediction"
— the *outer* task loss, not a surrogate internal to the layer. Read literally,
Eq. 71 therefore describes a **gradient-accumulation schedule over optimiser
steps**: level ℓ sums the task gradient for `C^(ℓ)` steps and applies it once.
That is what `ContinuumMemorySystem.pre_optimizer_step` implements, exactly, and
it is consistent with the M3 optimiser of §7.2 (same idea applied to momentum
across optimiser steps) and with §7.3's proposal to initialise CMS blocks from
pre-trained MLP weights. The paper also gestures at a within-sequence reading
(`C^(ℓ) = L/f`, "sequence parallelization"); the two are not equivalent, and we
took the unambiguous one.

A third, smaller call: Eq. 90 indexes the retrieval memory by `C·⌈t/C⌉`, which
is not causal for `t` inside a chunk. We use the chunk *start* (floor), which is.

## Implementation notes

* **Linear (matrix-valued) memories with residual retrieval** — `x + Mx`, not
  `Mx`. §8.2 derives the closed-form recurrence for exactly this case (Eq. 93),
  and it is the form whose inner gradient is exact without calling autograd
  inside the recurrence. The residual structure of Eq. 89 is preserved, so the
  meta-learned initial state is a sensible starting point rather than a
  degenerate one.
* **The chunk is semantic, not an optimisation.** §8.2 computes `k, v, η, α`,
  the generated values, and the gradients from the memory state at the *start*
  of the chunk. That changes what the model computes, so it is implemented
  faithfully and exposed as `chunk_size`; `chunk_size=1` recovers the online
  form of Eqs. 86–88.
* **One code path for `forward` and `step`.** The paper's parallel-within-chunk
  dual form is a speed optimisation, and at L ≤ 69 it buys nothing. A separate
  fast path is the classic way to end up with a model whose training-time and
  rollout-time behaviour silently differ, so both entry points call the same
  `_advance` and `test_backbone_forward_matches_step` holds by construction
  (verified: max abs difference 0.0).
* **η and α are bounded** (`eta_max`, `alpha_min`). Unbounded, the recurrence
  diverges within a few hundred steps at this scale; the paper's experiments run
  at a scale where the surrounding normalisation absorbs it. Both are logged
  every epoch via `extra_metrics()`.
* **L2-normalised q and k**, and a depthwise causal convolution of window 4,
  per §8.3.

## What the paper actually offers as "memory"

The paper contributes three separable memory mechanisms. They are genuinely
different objects, and picking between them changes the implementation, the
parameter budget, and which of our benchmarks should move. In the paper they
are combined, but nothing forces us to take all three.

### Candidate A — Continuum Memory System (CMS), §7.1, Eqs. 70–71

A chain of MLP blocks, each with its own **update frequency**:

```
y_t = MLP^(f_k)( MLP^(f_{k-1})( … MLP^(f_1)(x_t) ) )
```

Block ℓ has chunk size `C^(ℓ)` and its parameters are updated *only* every
`C^(ℓ)` steps, from the accumulated error over that chunk:

```
θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i − Σ_{t=i−C^(ℓ)}^{i} η^(ℓ)_t · f(θ^(f_ℓ)_t ; x_t)   if i ≡ 0 (mod C^(ℓ))
θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i                                                     otherwise
```

The claim: a standard Transformer block is the degenerate case `k = 1` with
update frequency zero. Spreading knowledge over blocks that update at different
rates is what resists catastrophic forgetting — when a fast block is overwritten,
the slower blocks still hold the old knowledge (§7.1, "CMS Design Helps with
Continual Learning").

Three connection variants are given: **nested** (Eq. 72), **sequential**
(Eq. 73), **independent / head-wise with an aggregator** (Eq. 74).

*What it would be here:* a replacement for the recurrent core in which the
"state" is a stack of MLP weight tensors updated at different cadences during the
sequence. Deep, high-capacity, simple learning rule.

*Which of our benchmarks it should move:* the continual-learning benchmark
(`wmcore/bench/continual.py`) most of all, then length extrapolation.

---

### Candidate B — Self-modifying (self-referential) Titans, §8.1, Eqs. 83–91

An associative memory in which the projections are themselves memories that
learn in-context, and which **generates its own training targets**:

```
y_t   = M_memory,t−1(q_t)
k_t   = M_k,t−1(x_t)     v_t = M_v,t−1(x_t)     η_t = M_η,t−1(x_t)     α_t = M_α,t−1(x_t)
v̂_□,t = M_□,t−1(v_t)                      ← each memory generates its own value
M_□,t = M_□,t−1 (α_t I − η_t k_t k_tᵀ) − η_t ∇L(M_□,t−1 ; k_t , v̂_□,t)
```

for `□ ∈ {k, v, q, η, α, memory}`, with `L(M;k,v) = ‖M(k) − v‖²`, each memory a
2-layer residual MLP `M(·) = (·) + W₁ σ(W₂ ·)`, and `q_t = x_t W_q` the only
non-adaptive projection. `η_t` is a data-dependent learning rate and `α_t` a
data-dependent retention (forget) gate. Initial states `M_□,0` are meta-learned
across sequences — i.e. they are ordinary trainable parameters of our model.

For the linear-memory case the update collapses to the closed form in Eq. 93:

```
M_□,t = M_□,t−1 (α_t I − η_t k_t k_tᵀ) − η_t (M_□,⌈t/C⌉·C k_t − v̂_□,t) k_tᵀ
```

Chunk-wise training (§8.2): with chunk size `C`, all of `k, v, η, α` and all
gradients for a chunk are computed from the state at the *end of the previous
chunk*, so a whole chunk is parallel. That is what makes this trainable at all.

*What it would be here:* the recurrent core becomes a set of fast-weight
matrices/MLPs updated by an inner gradient step at every timestep. Small
capacity, very expressive learning rule.

*Which of our benchmarks it should move:* the retention probes and the
memory-critical NLL — this is the mechanism aimed at in-context recall.

---

### Candidate C — HOPE, §8.3, Eqs. 94–97 — **self-modifying Titans followed by CMS**

The paper's actual proposed architecture: run the self-modifying Titans block,
then feed its output `o_t` through the CMS chain.

```
o_t = M_memory,t−1(q_t)          (Titans block, Eqs. 94–96)
y_t = MLP^(f_k)( … MLP^(f_1)(o_t) )     (CMS chain, Eq. 97)
```

Their stated rationale (§8.3): CMS has large capacity but a simple learning rule;
self-modifying Titans has small capacity but an expressive learning rule; they
are complementary. They also L2-normalise q and k and use local convolutions of
window size 4. A `Hope-Attention` variant swaps the Titans block for softmax
attention.

---

## Recommendation

**Take Candidate C (HOPE) as the headline, and build it out of A and B as
separately switchable parts.**

Reasoning:

1. HOPE is what the paper proposes and what a reviewer will expect "we used the
   memory from the nested learning paper" to mean.
2. Because HOPE *is* A composed with B, implementing it as two switchable
   sub-modules gets us the ablation table for free — `supreme` (full),
   `supreme-cms` (A only), `supreme-titans` (B only) — all four columns
   including the LSTM baseline, from one implementation, with the existing
   registry and no changes anywhere else in the repo.
3. Our two benchmark families map cleanly onto the two mechanisms (retention
   probes ↔ Titans, continual learning ↔ CMS), so an ablation actually
   *explains* the result instead of just reporting it.

**Confirm before I build:** which of A / B / C you meant. If you meant one
mechanism only, say which, and I'll implement just that — the rest of the
repository does not change either way.

---

## Implementation constraints, decided in advance

These are fixed by the apples-to-apples protocol and are not negotiable per
model:

| Constraint | Value | Why |
|---|---|---|
| Output head | shared `MDNHead`, 5 mixtures | isolates the recurrent core |
| `feature_dim` | must equal `memory.hidden_dim` (256) | C's input size, and therefore the CMA-ES search dimension, has to match |
| Parameter budget | within 10% of the LSTM backbone | otherwise the claim is "bigger", not "better" |
| Loss, optimiser, LR, epochs, clipping, seeds | from the shared config | one axis of variation |
| `critical_loss_weight` | 20, same as the baseline | see `docs/findings.md` §1 -- at weight 1 every layer scores at chance |
| Must run in < ~2 GB on MPS at batch 32 × length 128 | laptop budget | see `wmcore/utils/device.py` |

The LSTM baseline backbone at `hidden_dim=256` with a 35-d input is
**~299 k parameters**; match that.

Practical notes for whoever writes the code:

* The chunk-wise formulation (§8.2) is not an optimisation, it is a requirement —
  a per-timestep Python loop over an inner gradient step will be ~100× slower
  than the LSTM and the comparison becomes untrainable on a laptop. Use chunk
  size `C` as a config knob and implement the parallel-within-chunk form.
* The inner gradient `∇L(M; k, v̂)` for the 2-layer residual MLP memory can be
  written in closed form; do not call `torch.autograd.grad` inside the recurrence
  or the graph will explode.
* `state_features(state)` must return a fixed `(B, 256)` vector. For a
  fast-weight memory the natural choice is `M_memory,t(q_t)` — the retrieved
  content, not the raw weights — because that is what a linear controller can
  actually use.
* MPS has no fused kernels for any of this; expect the backbone to be the
  wall-clock bottleneck and report it in the cost table.
* Two tests in `tests/` are the acceptance criteria for the new backbone and
  will catch the mistakes that are otherwise invisible:
  `test_backbone_forward_matches_step` (the sequence pass and the single-step
  pass must agree to 1e-5 -- a backbone that fails this trains fine and is
  silently wrong in every open-loop, dream and control result) and
  `test_backbone_is_causal`. Add `"supreme"` to `ALL_BACKBONES` in
  `tests/test_core.py` and both run automatically.
