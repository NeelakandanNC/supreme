# World Model study

Two things live here:

* **`World Model/`** — the complete, runnable project: a laptop-scale
  reproduction of Ha & Schmidhuber's *World Models* (V + M + C), plus the shared
  benchmark harness, plus the slot where the Supreme memory layer plugs in.
  Start at [`World Model/README.md`](World%20Model/README.md).

The source papers sit one level up: `world_model.pdf` (Ha & Schmidhuber 2018)
and `nested_learning.pdf` (Behrouz et al., NeurIPS 2025).

## The experiment in one paragraph

Build a world model that trains end to end on an M4 MacBook. Keep V (a ConvVAE),
C (a linear controller trained with CMA-ES), the environments, the data, the
output head, the loss and the seeds fixed. Swap **only** M's recurrent core:
the LSTM of the original MDN-RNN versus a memory taken from the nested-learning
paper ("Supreme"). Because the memory layer is the single axis of variation,
benchmark it with memory benchmarks — retention across delay, capacity,
imagination horizon, continual-learning forgetting — rather than with generic
control scores.

## Status

| Piece | State |
|---|---|
| Shared core (`wmcore`) | done |
| Baseline memory: LSTM / MDN-RNN | done |
| Memory-stress environments, benchmark suite, reporting | done |
| Supreme memory core: Hope (self-modifying Titans → Continuum Memory System) | done, with `supreme-titans` and `supreme-cms` ablations |

## Verified

* 23 correctness tests pass (`make test`), including the contract every new
  memory layer must satisfy: the sequence pass and the single-step pass must
  agree, or the model you evaluate is not the model you trained.
* The suite discriminates: on `TMaze-v0` at corridor 16, an LSTM core and a GRU
  core — identical in everything else — separate at every level of the
  benchmark, from probe retention (2 vs 16 steps) through open-loop error to
  real-environment return (0.25 vs 1.00), while their average one-step NLL is
  nearly the same. Numbers in [`docs/findings.md`](World%20Model/docs/findings.md).
* One finding needs to go in the paper regardless of which memory wins: the
  plain world-model objective gives a recurrent core almost no incentive to
  retain task-critical information (~0.7 nats out of ~25). Measured, quantified,
  and corrected for identically in both arms.

Full A/B for both models, including CMA-ES, runs in about 20 minutes on the M4.
