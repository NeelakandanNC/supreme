# Memory-layer comparison -- `CarRacing-v0`

Every number below comes from runs whose configs differ **only** in `memory.backbone` (verified by `wmcore.config.assert_comparable`). V, C, the dataset, the MDN head, the optimiser and the seeds are shared.

**Settings.** env `CarRacing-v0` {}; train window 64; seeds [0]; device `mps` on arm64, torch 2.13.0.

Note the `memory.critical_loss_weight` used for these runs (see each run's `config.json`, and `docs/findings.md` §1 for why it matters): at weight 1.0 every memory layer scores at chance on the decision metric.

## Headline results

| Metric | attention | gru | lstm | supreme | Better |
|---|---|---|---|---|---|
| Decision accuracy (model) | -- | -- | -- | -- | higher |
| 1-step NLL | 10.19 | 11.13 | 10.87 | 10.36 | lower |
| NLL @ memory-critical | -- | -- | -- | -- | lower |
| NLL @ long eval length | 10.18 | 10.78 | 10.48 | 10.29 | lower |
| Open-loop MSE at step 10 | 13.51 | 8.73 | 7.864 | 9.411 | lower |
| Open-loop MSE at step 20 | 15.46 | 10.19 | 9.999 | 12.7 | lower |
| Open-loop MSE at step 50 | 17.01 | 11.68 | 11.19 | 14.31 | lower |
| Open-loop MSE at step 100 | 20.58 | 16.55 | 17.11 | 19.51 | lower |
| Drift slope | 0.008318 | 0.008874 | 0.009554 | 0.01109 | lower |
| Probe AUC (state) | -- | -- | -- | -- | higher |
| Probe AUC (obs control) | -- | -- | -- | -- | higher |
| Effective horizon | 0 | 0 | 0 | 0 | higher |
| Return (real env) | -- | -- | -- | -- | higher |
| Decision accuracy (agent) | -- | -- | -- | -- | higher |
| Forgetting on T0 (acc) | -- | -- | -- | -- | lower |
| Backbone params | 2.993e+05 | 2.25e+05 | 3e+05 | 3.01e+05 | lower |
| Step latency (ms) | 1.24 | 0.5229 | 0.473 | 4.385 | lower |
| Train pass (ms) | 2.027 | 6.871 | 1.99 | 1014 | lower |

_Seeds per model: attention=1, gru=1, lstm=1, supreme=1._


## Open-loop imagination error by horizon

| mse_at_horizon | attention | gru | lstm | supreme |
|---|---|---|---|---|
| 1 | 2.548 | 2.071 | 1.832 | 1.854 |
| 2 | 3.378 | 2.466 | 2.151 | 2.169 |
| 5 | 6.905 | 4.769 | 4.325 | 4.383 |
| 10 | 13.51 | 8.73 | 7.864 | 9.411 |
| 20 | 15.46 | 10.19 | 9.999 | 12.7 |
| 50 | 17.01 | 11.68 | 11.19 | 14.31 |
| 100 | 20.58 | 16.55 | 17.11 | 19.51 |


## Cue retention by delay (linear probe accuracy)

`state` probes the recurrent features h_t; `obs` probes the latent z_t alone and is the control -- once the cue leaves the screen it falls to chance, so any gap above it is memory.



## Cost

Measured on the machine named in each run's `meta.device`.
