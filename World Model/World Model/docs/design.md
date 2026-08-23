# Design notes: how the comparison is kept honest

The whole repository is arranged around one sentence:

> The baseline and Supreme differ in **exactly one** component — the recurrent
> core of the M model — and in nothing else.

Everything below follows from that.

## 1. One shared core, two thin model packages

```
wmcore/          SHARED: envs, data, V, MDN head, loss, C, dream, benchmarks
World Model/     MODEL 1: the LSTM recurrent core   (Ha & Schmidhuber 2018)
Supreme/         MODEL 2: the nested-learning core  (this work)
```

**Why the two models are two small folders and not two projects.** The obvious
layout would be a self-contained copy of the pipeline per model. That layout
cannot support the claim this study makes. The moment there are two copies of
the data loader, the VAE, the loss and the controller, the two arms can drift —
a different default here, a stale fix there — and every reported difference
becomes "the memory layer, plus whatever else diverged". Nothing in a diff of
two 60-file trees will tell you which.

With one shared core there is exactly one axis of variation and it is the axis
the paper is about. A model folder contributes a single class and one
`register_backbone(...)` call; it *cannot* touch the data pipeline or the
training loop, because it has no code there to touch. The ablation is enforced
by the layout of the code rather than by the discipline of whoever runs the
experiments — and `assert_comparable()` (§5) mechanically checks the configs on
top of that.

Model folders are discovered by path, not by import name, which is why they can
be called `World Model/` and `Supreme/` — with the space — rather than something
Python would let you type in an `import` statement.

## 2. The output head is shared, not per-model

M = **backbone** + **head**. The head is a 5-component diagonal Mixture Density
Network, identical for every backbone, defined once in `wmcore/memory/base.py`.

This matters more than it looks. If the baseline had an MDN head and the
challenger had, say, a Gaussian or a discretised head, then a difference in NLL
would confound *what the model remembers* with *how it expresses uncertainty*,
and no ablation in the paper could separate them. Fixing the head makes an NLL
difference attributable to the recurrent core by construction.

The same argument applies to the reward and termination heads, the loss, the
masking, the optimiser, the schedule and the gradient clipping — all shared.

## 3. V is trained once and frozen

The ConvVAE is trained per *environment*, stored next to the *dataset*, and both
models load the identical checkpoint (`wmcore/train/train_vae.py`,
`vision.checkpoint`). Latents are precomputed once
(`wmcore/train/encode_latents.py`).

Two reasons. Scientifically, retraining V per model would let VAE seed noise
leak into the memory comparison, and that noise is larger than the effect we are
chasing. Practically, precomputing latents turns M-training from a 1.5 GB
image problem into a 30 MB vector problem, which is what makes many seeds and
many sequence lengths affordable on a laptop.

## 4. C is linear, and identical

The controller is a single linear map from `[z_t ; h_t]` to an action —
under a thousand parameters. Ha & Schmidhuber keep C trivial so that competence
must live in V and M; here that argument does double duty. Since C is linear and
identical in both legs, a difference in return *is* a difference in how much
task-relevant information the memory state exposes **linearly** — which is also
exactly what the linear probes in `wmcore/bench/probes.py` measure.

`controller.hidden_input: false` gives the no-memory ablation (C sees `z_t`
only). Reporting it is the cheapest way to show the memory is doing anything at
all.

## 5. Machine-checked comparability

`wmcore.config.assert_comparable(cfg_a, cfg_b)` diffs two configs and raises
unless the only differing keys are `name`, `memory.backbone` and
`memory.backbone_kwargs`. `wm.py study` calls it before training. If someone
changes the learning rate for one model, the study aborts instead of producing a
publishable-looking number.

## 6. Parameter budgets are reported, and matched

`wmcore/bench/cost.py` reports total, backbone and head parameter counts,
per-step inference latency, training-pass time and peak RSS for every run.
"Our layer is better" and "our layer is bigger" are different papers; the cost
table is what tells them apart. The LSTM backbone at `hidden_dim = 256` is
~300 k parameters — match it within ~10%.

## 7. Seeds

Nothing here is single-seed. `make sweep SEEDS="0 1 2"` runs the full A/B per
seed and `wm.py report` aggregates mean ± s.e.m. CMA-ES in particular has
between-seed variance comparable to any plausible effect size, which is precisely
why the synthetic memory suite — not CarRacing — carries the argument.

## 8. Why the environments are synthetic

CarRacing is included for external comparability and is *not* the discriminating
benchmark: it is close to fully observable from a single frame, so the memory
layer has little to do.

The synthetic suite (`wmcore/envs/memory_envs.py`) is built so that a single cue
presented at t=0 is required to predict a frame arriving `delay` steps later,
with `delay` a free parameter. That gives the study its independent variable — a
memory-horizon axis you can sweep in minutes — and it puts the memory requirement
inside the *world-model loss*, not only in the reward: the environment tags the
one transition per episode that is unpredictable without the cue
(`memory_critical`), and the benchmark reports NLL restricted to those steps.

## 9. One deliberate departure from a pure ablation

`memory.critical_loss_weight` (default 20 in the memory configs) upweights the
transitions the environment flags as requiring memory.

This is the only place where the objective is not the textbook world-model
objective, and it is here because the textbook objective turns out not to test
what we need it to. Measured on this repository: at weight 1.0 the memory-
critical transition carries ~1/L of the loss and getting it right saves ~0.7
nats out of ~25, so a converged LSTM predicts the midpoint of the two possible
successors and scores at chance — while a probe shows the cue *was* in its state
earlier in the corridor. Every memory layer sits at chance and the benchmark has
zero headroom. Full numbers in [findings.md](findings.md) §1.

The weight is applied identically to every backbone, changes no architecture,
and leaves the *reported* NLL unweighted so numbers stay comparable across
settings. It also lowers unweighted NLL, so it is not buying decision accuracy
at the expense of model quality. Report it in the paper, and report the weight-1
ablation next to it.

## 10. Hardware envelope

Target: Apple M4, 16 GB unified memory, ~10 GB usable.

* Frames live in a memory-mapped `.npy`; the OS page cache handles residency.
* Latents are precomputed, so M-training is RAM-trivial.
* `MemoryBudget` trips at 9.5 GB RSS with an actionable message instead of
  letting the machine swap.
* CMA-ES workers run on **CPU**, not MPS: six Metal contexts cost more than they
  save for models this small, and it keeps the GPU free.
* Default configs are sized so a full single-seed study on `configs/tmaze.yaml`
  finishes in well under an hour.
