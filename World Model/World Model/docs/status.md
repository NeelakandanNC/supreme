# Benchmark status

Last updated after the corridor-32 headline (2 seeds) and CarRacing.

## Completed

| what | config | backbones | seeds | notes |
|---|---|---|---|---|
| Headline | `tmaze_hard.yaml` (corridor 32) | lstm, gru, supreme | 2 | control for lstm/gru both seeds, supreme seed 0 only |
| CarRacing | `carracing.yaml` (half scale) | lstm, gru, attention, supreme | 1 | no controller |
| Corridor 16 | `tmaze.yaml` | lstm, gru | 1 | Supreme columns there are stale, see below |
| lambda sweep | corridor 6 | lstm | 1 | the objective-incentive measurement |

Headline numbers (mean +- s.e.m. over 2 seeds):

| | lstm | gru | **supreme** |
|---|---|---|---|
| Decision accuracy (chance 0.5) | 0.475 +- 0.047 | 0.483 +- 0.006 | **1.000 +- 0.000** |
| NLL @ memory-critical | 25.55 +- 2.15 | 25.30 +- 2.34 | **15.61 +- 0.97** |
| NLL, all steps | 14.08 +- 1.99 | 14.12 +- 1.97 | 13.68 +- 2.16 |
| Effective memory horizon | 4 | 20 +- 4 | **32** |
| Probe accuracy at delay 32 | 0.411 +- 0.078 | 0.533 +- 0.067 | **0.978 +- 0.000** |
| Real-env return | 0.06 +- 0.19 | -0.13 +- 0.13 | **1.00** (n=1) |

## Not run

Ordered by what it would cost the paper to omit.

1. **Ablations** -- `supreme-titans`, `supreme-cms`, `hope-attention`.
   Implemented, registered, parameter-accounted, never executed. These carry the
   sharper claim (*which half* of Hope does the work); without them the paper can
   only say the composition beats an LSTM.
   Run with `./scripts/run_ablations.sh`.
2. **A third seed.** Two is not a distribution, even when both agree exactly.
3. **`SequenceRecall-v0`** -- the capacity axis, complementary to retention.
4. **Continual learning** -- must be run at a corridor the model under test
   actually solves, or the metric is vacuous (the benchmark self-reports
   `informative: false`).
5. **Corridor sweep** -- converts the choice of corridor 32 from a selected
   operating point into a curve whose crossover is the finding.
6. **`critical_loss_weight = 1` at the operating point** -- currently measured at
   corridor 6 only.
7. **Dream-trained controller** -- code path verified end to end, never reported.

`scripts/run_deferred.sh` covers 3-6.

## Caveats attached to what exists

* **Supreme seed 1 trained for 60 of 150 epochs** before being interrupted. Its
  best-validation checkpoint was past the phase transition and reaches the same
  decision accuracy (1.000) and probe accuracy (0.978) as seed 0, so it is
  reported -- but the two seeds do not share a training budget, and the error bar
  on Supreme's NLL mixes the two.
* **Supreme's real-env return is a single seed** (seed 0); no error bar.
* **CarRacing ran at half scale** (150 rollouts, 12 memory epochs, no controller).
* **Stale Supreme results.** The Titans norm-projection change altered what
  `supreme` computes. Supreme columns in `runs/` (corridor 16) predate it and
  must not be quoted. `lstm` and `gru` are unaffected.
