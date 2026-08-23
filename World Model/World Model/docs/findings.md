# Measured baseline findings

Results produced by this repository with the **baseline** (LSTM/MDN-RNN) memory
core, on an Apple M4 / 16 GB. They are recorded here because two of them shape
how the Supreme comparison must be run and reported — and one of them is a
result in its own right.

Reproduce with the commands given under each heading.

---

## 1. The world-model objective barely rewards remembering

**This is the most important thing measured so far, and it belongs in the paper.**

In a delayed-recall episode exactly one transition cannot be predicted without
memory: the frame that reveals the answer. At sequence length *L* that
transition carries roughly *1/L* of the loss, and the most a model can gain by
getting it right is the entropy of the outcome — about **0.69 nats for a binary
cue, against a per-step NLL of ~25**.

The consequence is measurable and stark. `TMaze-v0`, corridor 6, 2000 rollouts,
60 epochs, LSTM core, `critical_loss_weight` swept:

| `critical_loss_weight` | Decision accuracy (chance 0.5) | NLL @ memory-critical | NLL (unweighted, all steps) | Probe accuracy at the decision step |
|---|---|---|---|---|
| 1   | **0.500** | 25.53 | 13.16 | 0.44 (chance) |
| 20  | **1.000** | 19.42 | 12.76 | 1.00 |
| 100 | **1.000** | 13.68 | 12.49 | 1.00 |

At weight 1 the converged model predicts almost exactly the *midpoint* of the
two possible successors (mean distance to the midpoint 0.25, against a
green–red centroid separation of 6.74). A linear probe confirms the cue was
held in the state four steps into the corridor and gone by the decision step.

So the model is not failing to remember. **It is correctly optimising an
objective that does not reward remembering**, and it spends its recurrent
capacity on the many locally-predictable corridor frames instead.

Two things follow:

1. **Report it.** "Maximum-likelihood world-model training gives a recurrent
   core almost no incentive to retain task-critical information, and we quantify
   the incentive at ~0.7 nats out of ~25" is a clean, defensible contribution
   independent of which memory layer wins.
2. **Correct for it before comparing memory layers.** At weight 1 every memory
   layer sits at chance and the benchmark has no headroom, so nothing can be
   distinguished. The configs therefore ship `critical_loss_weight: 20`, applied
   identically to both models and disclosed. Note that it also *lowers*
   unweighted NLL (13.16 → 12.76), so it is not a quality trade-off — it is a
   fix to an optimisation-priority pathology.

```bash
for W in 1 20 100; do
  python wm.py study -c configs/tmaze.yaml --backbones lstm --no-control \
    -o out_dir=runs_w$W -o memory.critical_loss_weight=$W
done
```

---

## 2. The suite discriminates cleanly, end to end

`configs/tmaze.yaml` as shipped (corridor 16, 1500 rollouts, 120 epochs,
`critical_loss_weight: 20`), one seed, LSTM versus GRU core, everything else
identical:

| | LSTM | GRU |
|---|---|---|
| **Decision accuracy** (chance 0.5) | 0.507 | **1.000** |
| NLL @ memory-critical | 27.53 | **15.33** |
| 1-step NLL (all steps) | 11.30 | **10.63** |
| Open-loop MSE at step 10 | 11.58 | **0.13** |
| Drift slope | 0.492 | **0.032** |
| Effective memory horizon | 2 | **16** |
| Probe accuracy at delay 16 | 0.554 | **1.000** |
| Real-env return (CMA-ES controller) | 0.25 | **1.00** |
| Agent decision accuracy | 0.625 | **1.000** |

The important property is not that the GRU wins — it is that **every level of
the suite agrees**, and they agree mechanistically:

* the probe says the LSTM state has lost the cue by delay 4 and the GRU still
  has it at delay 16;
* so the LSTM cannot predict the outcome frame, and its open-loop error explodes
  by exactly step 10 (where the decision transition falls) while the GRU's stays
  flat;
* so the LSTM's controller cannot solve the maze (return 0.25) and the GRU's
  solves it outright (return 1.00).

A suite where a mechanistic claim about the state propagates all the way to
control is one that can support a causal claim in a paper. Note also that the
average one-step NLL barely separates them (11.30 vs 10.63) while the
memory-critical NLL separates by 12 nats — the point made in §3 below.

Run time for the whole comparison, both models, on the M4: about 20 minutes
including data collection, V, both M models, both CMA-ES searches and the full
benchmark.

```bash
python wm.py study -c configs/tmaze.yaml --backbones lstm,gru
python wm.py report runs
```

---

## 3. Supreme (Hope) vs. the baseline, with ablations

`configs/tmaze.yaml` (corridor 16, 1500 rollouts, 120 epochs,
`critical_loss_weight: 20`), one seed, everything except `memory.backbone`
identical:

| | lstm | **supreme** | supreme-titans | supreme-cms | gru |
|---|---|---|---|---|---|
| Decision accuracy (chance 0.5) | 0.507 | **1.000** | 1.000 | 0.556 | 1.000 |
| NLL @ memory-critical | 27.53 | 16.00 | **15.74** | 26.93 | 15.33 |
| 1-step NLL (all steps) | 11.30 | **9.70** | 11.09 | 10.85 | 10.63 |
| Effective memory horizon | 2 | **16** | 16 | 4 | 16 |
| Real-env return | 0.25 | **1.00** | 1.00 | 0.25 | 1.00 |
| Backbone params | 299,008 | 300,696 | 224,520 | 375,184 | 224,256 |
| Train pass (ms, B=64 × L=21) | 1.8 | 265 | 403 | 10.7 | 5.4 |

Three things to take from this, and two of them are inconvenient.

**The result is real.** Hope solves a task the MDN-RNN baseline cannot touch, at
matched parameters (+0.6%), with identical data, V, head, loss, optimiser and
seeds. Decision accuracy 0.507 → 1.000, memory horizon 2 → 16, real-environment
return 0.25 → 1.00.

**The ablation says the win is entirely the Titans half, not CMS.**
`supreme-titans` (Hope with CMS removed, 25% *fewer* parameters) matches full
Hope on every memory metric — decision accuracy 1.000, horizon 16, return 1.00.
`supreme-cms` (the baseline LSTM with CMS bolted on, 25% *more* parameters) is
essentially the baseline: 0.556, horizon 4, return 0.25. The self-modifying
memory does the remembering; the Continuum Memory System does not.

CMS is not useless, though — it is doing something different. Full Hope has the
best overall one-step NLL (9.70) and `supreme-titans` the worst of the two
(11.09), so CMS improves general predictive quality by about 1.4 nats while
contributing nothing to retention. That is a coherent, reportable division of
labour, and it is exactly the claim §8.3 makes about the two halves being
complementary — just with a sharper split than the paper implies.

**At this operating point the benchmark cannot separate Hope from a GRU.** The
GRU matches Hope on every memory metric with 25% fewer parameters and ~50× less
wall-clock. Corridor 16 saturates: three of the five models sit at the ceiling.
A paper cannot claim a memory advantage from a saturated benchmark, so the
comparison has to move to an operating point where the ceiling is not reached —
a longer corridor (`configs/tmaze_hard.yaml`, corridor 32;
`configs/tmaze_long.yaml`, corridor 64) or the capacity task
(`configs/recall.yaml`, four simultaneous associations), which is where an
associative memory should have an advantage a gated RNN does not.

**Cost is an implementation artefact, and should be labelled as one.** Hope's
265 ms training pass against the LSTM's 1.8 ms is the cost of a sequential
Python recurrence, not of extra learning capacity: both models see the same
data for the same number of epochs with the same parameter count. §8.2's
parallel-within-chunk dual form would remove most of it. Report wall-clock, but
do not let a reviewer read it as "spent more compute to learn more".

**One anomaly, unexplained.** The retention probe for `supreme` and
`supreme-titans` is 1.00 at delays 0–2, dips to ~0.50 at delay 4, then returns
to 1.00 at delays 8, 12 and 16. It is non-monotonic, which retention curves
should not be. Candidate causes: an interaction with the Titans chunk boundary
(`chunk_size=8`), or probe variance at a single delay with ~28 test episodes.
Worth resolving before the curve goes in a figure — run more seeds and sweep
`chunk_size`.

---

## 4. Corridor 32 — the headline result

`configs/tmaze_hard.yaml` (corridor 32, 1200 rollouts, `critical_loss_weight: 20`),
one seed. Two budgets, because the difference between them is itself the finding:

**60 epochs** — retention separates, task performance does not:

| | lstm | gru | supreme |
|---|---|---|---|
| Probe accuracy at delay 32 | 0.311 | 0.444 | **0.867** |
| Decision accuracy | 0.506 | 0.528 | 0.528 |

**150 epochs** — Supreme converts the retention into task performance; neither RNN does:

| | lstm | gru | **supreme** |
|---|---|---|---|
| **Decision accuracy** (chance 0.5) | 0.522 | 0.489 | **1.000** |
| NLL @ memory-critical | 23.40 | 22.96 | **15.52** |
| 1-step NLL | 16.07 | 16.09 | **15.38** |
| Probe accuracy at delay 16 | 0.333 | 0.867 | **1.000** |
| Probe accuracy at delay 32 | 0.333 | 0.467 | **1.000** |
| Effective memory horizon | 4 | 16 | **32** |
| Backbone params | 299,008 | 224,256 | 300,696 |

**This is the result to build the paper on.** At a 32-step required horizon,
Hope solves the task outright while both the MDN-RNN baseline and a GRU control
sit at chance — and the probe explains exactly why: Hope's state still encodes
the cue perfectly at the decision point (1.000), the GRU's has decayed to chance
(0.467), the LSTM's is gone by delay 8.

It also retires the objection raised by the corridor-16 run (§3), where a GRU
matched Hope on every metric. That was a ceiling effect. Corridor 16 is
saturated; corridor 32 is not, and there the ordering is unambiguous:

    effective memory horizon:   LSTM 4  <  GRU 16  <  Hope 32

**The two budgets are worth reporting together.** At 60 epochs Hope already
*held* the cue (probe 0.867) but had not yet learned to *use* it (decision
accuracy at chance); by 150 epochs it had. Retention and exploitation are
separate milestones, and a study that stops at the first budget would have
reported "no difference in task performance" while the mechanism was already in
place. This is the same pattern as §1 and §2, in its sharpest form, and it is a
concrete methodological warning for anyone benchmarking memory layers.

---

## 5. The probe control behaves exactly as it must

`TMaze-v0`, corridor 24, LSTM and GRU cores, linear probe on the recurrent
state versus the observation-only control:

| delay since cue | 0 | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|
| GRU, state | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.48 | 0.48 |
| LSTM, state | 1.00 | 1.00 | 1.00 | 0.68 | 0.48 | 0.48 | 0.48 | 0.48 |
| observation control | 1.00 | 0.48 | 0.48 | 0.48 | 0.44 | 0.48 | 0.48 | 0.48 |

The control is at 1.00 while the cue is on screen and at chance from the very
next step, which is the sanity condition for the whole benchmark: anything above
it is memory and nothing else. And the two cores separate cleanly — effective
memory horizon 12 versus 4 — while their **average one-step NLL is identical to
three significant figures** (12.88 vs 12.87).

That last point is the case for this benchmark suite. A paper that reported only
world-model NLL would have concluded these two memory layers are the same.

---

## 6. Cost baseline

Backbone parameter counts at `hidden_dim=256` with a 34-d input
(32-d latent + 2 actions):

| core | backbone params | full M | step latency | train pass (B=32, L=128) |
|---|---|---|---|---|
| LSTM | 299,008 | 422,882 | ~0.5 ms | ~1.2 ms |
| GRU | 224,256 | 348,130 | ~0.5 ms | ~2.2 ms |

The LSTM figure matches the count reported in Ha & Schmidhuber's appendix, and V
reproduces their 4,348,547 exactly. **Supreme's backbone should land within ~10%
of 299 k** or the comparison becomes a capacity comparison.

---

## 7. Continual learning needs a solvable operating point

Run on `configs/tmaze.yaml` (corridor 16) with the LSTM core, 3 task variants,
40 epochs each, the decision-accuracy matrix came out as:

```
     before: 0.411  0.589  0.411
   after T0: 0.411  0.589  0.411
   after T1: 0.589  0.411  0.589
   after T2: 0.411  0.589  0.411
```

`backward_transfer_accuracy = 0.089`, `forgetting_task0_accuracy = 0.000` — all
well-formed, all meaningless. The LSTM never learns *any* of the tasks at
corridor 16 (see §2), so the matrix is just its output bias reflected across the
variants: variant 1 is the swapped permutation, so a model that always predicts
one outcome scores 0.411 / 0.589 / 0.411 by construction.

The benchmark now sets `informative: false` and emits a warning in this case.
**Run the continual protocol at a corridor length where the layer under test
solves the single-task version** — corridor 6 for the LSTM, corridor 16 or more
for the GRU. Otherwise the number is an artefact.

---

## 8. What this means for the Supreme comparison

* Run at `critical_loss_weight: 20`, and report the weight-1 numbers as an
  ablation — the gap between them is itself informative about a memory layer.
* Lead with **decision accuracy** and **probe retention**, not average NLL.
  Average NLL is dominated by locally-predictable frames and does not move.
* **Do not report corridor 16 as the headline.** It is saturated: Hope, its
  Titans-only ablation and a plain GRU all hit the ceiling, so it shows Hope
  beats the MDN-RNN baseline but cannot show Hope beats a well-chosen RNN.
  Corridor 32 (`configs/tmaze_hard.yaml`) is where they separate — see §4 — and
  the corridor sweep (`make sweep-horizon`) is the figure that makes the point:
  the models tie at the easy end, and the GRU falls off between 16 and 32 while
  Hope does not. `configs/recall.yaml` (four simultaneous associations) is the
  complementary capacity axis.
* **Include the GRU column.** Leaving it out would make the result look stronger
  than it is, and a reviewer will ask.
* **Lead the ablation with `supreme-titans`.** It is the finding: the memory
  advantage comes from the self-modifying associative memory, and CMS buys
  general NLL rather than retention. That is a sharper claim than "Hope works"
  and it is better supported by the data.
* Before reporting continual learning, check `informative` in the result — see §7.
* Use the corridor-length sweep (`make sweep-horizon`) to find the regime where
  the baseline has begun to fail but is not yet at floor. That is where a better
  memory layer has room to show a difference, and it is the figure to build the
  paper around.
