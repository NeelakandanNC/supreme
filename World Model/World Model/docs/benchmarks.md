# The benchmark suite

Run with `python wm.py bench -c <config>`; results land in
`runs/<name>/seed<k>/bench.json`. `python wm.py report runs` aggregates them.

Because the two models differ only in the memory layer, every metric here is a
*memory* metric by construction. They are ordered from cheapest and most direct
to most end-to-end.

---

## 0. Decision accuracy — `decision`  **(the headline metric)**

At the one transition per episode that requires memory, does M predict the frame
that actually comes next?

The two possible successors — a green "correct" frame and a red "wrong" frame —
occupy well-separated regions of latent space. We estimate their centroids from
ground-truth data, take the model's predicted next latent, assign it to the
nearer centroid, and check it against the truth.

Why this leads rather than NLL:

* **Chance is exactly 0.5**, so the scale is interpretable with no baseline model.
* **It cannot be gamed by hedging.** A model that predicts the average of the
  two outcomes lands between the centroids and scores at chance — and hedging is
  precisely the NLL-optimal strategy for a model that has not learned the rule
  (see [findings.md](findings.md) §1).
* **It requires memory by construction**: the outcome is a function of the cue
  (seen at t=0) and the action (taken now); nothing in the current frame reveals it.
* **It is measured on the model, not through a controller**, so it carries none
  of the CMA-ES variance.

`margin` is reported alongside: the centroid separation relative to the spread of
the data. If it is small the metric is meaningless and V, not the memory layer,
is the problem.

> **Read this together with `memory.critical_loss_weight`.** At weight 1.0 every
> memory layer scores at chance here, because the objective barely rewards
> getting this step right. The configs ship weight 20. See
> [findings.md](findings.md) §1 for the measurement and the justification.

---

## 1. Teacher-forced prediction — `teacher_forced`

One-step NLL and MSE of `p(z_{t+1} | z_{≤t}, a_{≤t})` with ground truth fed at
every step.

* `nll_1step` — the standard world-model number. Expect the two models to be
  close: the average is dominated by locally predictable frames.
* **`nll_memory_critical`** — the same NLL restricted to the transitions the
  environment flagged as requiring the cue. On the synthetic suite this is the
  cleanest single read-out of the whole study. If a memory layer helps at all,
  it shows up here first and largest.

## 2. Length extrapolation — `length_extrapolation`

The identical metric evaluated on windows longer than anything seen in training
(`data.eval_sequence_length` vs `data.sequence_length`).

**On the delayed-recall environments this is reported as not applicable**, and
that is a deliberate consequence of a hard constraint: a training window must
span cue → decision, which is the whole episode, so there is no shorter training
window to extrapolate from. `check_window_spans_the_dependency()` enforces the
constraint and will abort a run that violates it — a shorter window does not
make the task harder, it removes the dependency from the training signal
altogether and every memory layer then scores at chance for reasons unrelated to
memory.

Extrapolation on these environments is therefore measured by **varying the
corridor length** (`scripts/sweep_horizon.py`), which is the more meaningful
axis anyway: it asks how far the memory reaches, not how long a window is.

## 3. Open-loop imagination — `imagination`, `imagination_sampled`

Warm the state on `context` ground-truth steps, then roll forward feeding the
model its own predictions back, using the true action sequence. Error is
reported per step and aggregated at horizons 1…N.

Two failure modes are separated, and a memory layer governs the second:

* error at small *k* measures local dynamics — both models should tie;
* the **slope** of the curve (`drift_slope`) measures whether the state retains
  enough to stay on the true trajectory.

`imagination_sampled` uses the temperature-τ mixture sampler, i.e. the dream the
controller actually experiences.

## 4. Retention probes — `probes`

The most direct measurement in the suite. Harvest the recurrent features `h_t`
over validation sequences; for each cue-to-probe delay `d`, fit a **linear**
ridge classifier from `h_t` to the ground-truth cue, split **by episode**.

* `probe_h` — accuracy from the memory state.
* `probe_z` — the control: the same probe on the latent `z_t` alone, i.e. what
  is visible in the current frame. It collapses to chance once the cue leaves
  the screen; anything above it past that point is memory.
* `effective_memory_horizon` — the largest delay where `probe_h` beats both
  chance and `probe_z` by >10 points. One number for the abstract.
* `auc_h`, `auc_z` — mean accuracy across delays.

Linear probes, not MLPs, on purpose: a nonlinear probe can manufacture accuracy
from a state no linear controller could use, and C here *is* linear, so the
probe measures the thing that matters downstream.

Episode-level splitting is not optional: consecutive timesteps share a label and
a near-identical state, so a random split reports ~100% for any model.

## 5. Continual learning — `continual`

Task-incremental protocol over environment variants that share pixel statistics
and differ only in the cue→action mapping. M is trained on each task in turn,
with no replay and no task label; after task *i* we evaluate on all tasks *j ≤ i*.

Reported in the style of Lopez-Paz & Ranzato (2017), with the sign convention
flipped because NLL is a loss (lower is better, and forgetting means NLL *rose*):

* `final_nll_mean` — mean NLL over all tasks after the last one;
* `backward_transfer_nll` — mean change on earlier tasks caused by later
  training; **positive means forgetting**;
* `forgetting_task0_accuracy` — the headline: how much decision accuracy on T₀
  was lost. Positive means forgetting.
* NLL versions of the same quantities are reported as `*_nll` secondaries.

Decision accuracy, not NLL, is the primary continual metric here — and that is a
correction, not a preference. The task variants share pixel statistics and
differ only in the cue→outcome rule, which touches one transition in ~30, so
average NLL is nearly identical across variants and shows no forgetting even
when the rule has plainly been overwritten. Measured: NLL per task after each
task was `20.111 / 20.054 / 20.111` — three tasks, indistinguishable.

V is frozen across tasks, which confines forgetting to the memory layer.

**Precondition, checked automatically.** Forgetting is only measurable if there
was something to forget. The result carries `informative: false` and a `note`
when the mean accuracy on each task *immediately after training it* is near
chance — measured: an LSTM core at corridor 16 cannot solve the single-task
version at all, and its continual matrix is nothing but its output bias
reflected across the variants, while still yielding a perfectly well-formed
BWT figure. Run this benchmark at a corridor length where the layer under test
does solve one task, or with a layer that can.

This is the benchmark the nested-learning line of work targets most directly, so
a nested-learning memory has to be measured on it and not only on likelihood.

## 6. Control — `control`

`return_mean` and `decision_accuracy` from real-environment episodes with the
CMA-ES-trained controller. Two variants worth running:

* `controller.hidden_input: false` — C sees `z_t` only. The gap to the full
  model is how much the memory contributes at all.
* `controller.train_in_dream: true` — C is trained entirely inside M's
  imagination and only evaluated for real. This is the most sensitive
  end-to-end probe of a memory layer: the controller can only *learn* a
  memory-dependent policy if the dream remembers, and can only *transfer* if
  what the dream remembers is true.

## 7. Cost — `cost`

Parameters (total / backbone / head), per-step inference latency, training-pass
time, throughput, peak RSS, device. Always reported next to the quality metrics:
the standing alternative hypothesis for any improvement is "you spent more
compute", and this table is what rules it out.

---

## Suggested table for the paper

| | LSTM (baseline) | Supreme |
|---|---|---|
| Decision accuracy (chance 0.5) | | |
| NLL @ memory-critical | | |
| Effective memory horizon (steps) | | |
| Open-loop MSE @ 100 | | |
| Forgetting on T₀ (ΔNLL) | | |
| Decision accuracy (real env) | | |
| Backbone params | | |
| Step latency (ms) | | |

plus two figures — open-loop divergence and the retention curve — both emitted
by `python wm.py report runs`.
