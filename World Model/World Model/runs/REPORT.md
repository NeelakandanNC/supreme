# Memory-layer comparison -- `TMaze-v0`

Every number below comes from runs whose configs differ **only** in `memory.backbone` (verified by `wmcore.config.assert_comparable`). V, C, the dataset, the MDN head, the optimiser and the seeds are shared.

**Settings.** env `TMaze-v0` {'corridor_length': 16, 'n_cues': 2, 'distractors': 3, 'cue_steps': 2, 'feedback_steps': 2}; train window 21; seeds [0]; device `mps` on arm64, torch 2.13.0.

Note the `memory.critical_loss_weight` used for these runs (see each run's `config.json`, and `docs/findings.md` §1 for why it matters): at weight 1.0 every memory layer scores at chance on the decision metric.

## Headline results

| Metric | gru | lstm | supreme | supreme-cms | supreme-titans | Better |
|---|---|---|---|---|---|---|
| Decision accuracy (model) | 1 | 0.5067 | 1 | 0.5556 | 1 | higher |
| 1-step NLL | 10.63 | 11.3 | 9.704 | 10.85 | 11.09 | lower |
| NLL @ memory-critical | 15.33 | 27.53 | 16 | 26.93 | 15.74 | lower |
| NLL @ long eval length | 10.63 | -- | -- | -- | -- | lower |
| Open-loop MSE at step 10 | 0.1284 | 11.58 | 0.5636 | 11.55 | 0.09792 | lower |
| Open-loop MSE at step 20 | -- | -- | -- | -- | -- | lower |
| Open-loop MSE at step 50 | -- | -- | -- | -- | -- | lower |
| Drift slope | 0.03156 | 0.4918 | 0.0254 | 0.4986 | 0.008462 | lower |
| Probe AUC (state) | 1 | 0.7474 | 0.926 | 0.7781 | 0.9184 | higher |
| Probe AUC (obs control) | 0.619 | 0.6148 | 0.6148 | 0.6148 | 0.6148 | higher |
| Effective horizon | 16 | 2 | 16 | 4 | 16 | higher |
| Return (real env) | 1 | 0.25 | 1 | 0.25 | 1 | higher |
| Decision accuracy (agent) | 1 | 0.625 | 1 | 0.625 | 1 | higher |
| Forgetting on T0 (acc) | -- | -- | -- | -- | -- | lower |
| Backbone params | 2.243e+05 | 2.99e+05 | 3.007e+05 | 3.752e+05 | 2.245e+05 | lower |
| Step latency (ms) | 0.9057 | 0.4654 | 2.478 | 1.562 | 4.777 | lower |
| Train pass (ms) | 5.432 | 1.811 | 265.4 | 10.69 | 403.3 | lower |

_Seeds per model: gru=1, lstm=1, supreme=1, supreme-cms=1, supreme-titans=1._


## Open-loop imagination error by horizon

| mse_at_horizon | gru | lstm | supreme | supreme-cms | supreme-titans |
|---|---|---|---|---|---|
| 1 | 0.06396 | 0.05241 | 0.2646 | 0.05733 | 0.08067 |
| 2 | 0.07177 | 0.06497 | 0.3009 | 0.06495 | 0.1036 |
| 4 | -- | 0.1079 | 0.399 | 0.1151 | 0.148 |
| 5 | 0.1022 | -- | -- | -- | -- |
| 6 | -- | 0.1267 | 0.3254 | 0.1421 | 0.163 |
| 8 | -- | 0.02056 | 0.3251 | 0.03487 | 0.06802 |
| 10 | 0.1284 | 11.58 | 0.5636 | 11.55 | 0.09792 |
| 20 | -- | -- | -- | -- | -- |
| 35 | -- | -- | -- | -- | -- |
| 50 | -- | -- | -- | -- | -- |


## Cue retention by delay (linear probe accuracy)

`state` probes the recurrent features h_t; `obs` probes the latent z_t alone and is the control -- once the cue leaves the screen it falls to chance, so any gap above it is memory.

| delay | gru (state) | gru (obs) | lstm (state) | lstm (obs) | supreme (state) | supreme (obs) | supreme-cms (state) | supreme-cms (obs) | supreme-titans (state) | supreme-titans (obs) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 0.518 | 1.000 | 0.518 | 1.000 | 0.518 | 1.000 | 0.518 | 1.000 | 0.518 |
| 2 | 1.000 | 0.554 | 1.000 | 0.554 | 1.000 | 0.554 | 1.000 | 0.554 | 1.000 | 0.554 |
| 4 | 1.000 | 0.536 | 0.625 | 0.536 | 0.500 | 0.536 | 0.857 | 0.536 | 0.482 | 0.536 |
| 8 | 1.000 | 0.571 | 0.554 | 0.571 | 1.000 | 0.571 | 0.554 | 0.571 | 1.000 | 0.571 |
| 12 | -- | -- | 0.500 | 0.589 | 0.982 | 0.589 | 0.500 | 0.589 | 0.946 | 0.589 |
| 16 | 1.000 | 0.536 | 0.554 | 0.536 | 1.000 | 0.536 | 0.536 | 0.536 | 1.000 | 0.536 |


## Cost

Measured on the machine named in each run's `meta.device`.
