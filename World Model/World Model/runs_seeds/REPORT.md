# Memory-layer comparison -- `TMaze-v0`

Every number below comes from runs whose configs differ **only** in `memory.backbone` (verified by `wmcore.config.assert_comparable`). V, C, the dataset, the MDN head, the optimiser and the seeds are shared.

**Settings.** env `TMaze-v0` {'corridor_length': 32, 'n_cues': 2, 'distractors': 3, 'cue_steps': 2, 'feedback_steps': 2}; train window 37; seeds [0, 1]; device `mps` on arm64, torch 2.13.0.

Note the `memory.critical_loss_weight` used for these runs (see each run's `config.json`, and `docs/findings.md` §1 for why it matters): at weight 1.0 every memory layer scores at chance on the decision metric.

## Headline results

| Metric | gru | hope-attention | lstm | supreme | supreme-cms | Better |
|---|---|---|---|---|---|---|
| Decision accuracy (model) | 0.4833 ± 0.0039 | 1 ± 0 | 0.475 ± 0.033 | 1 ± 0 | 0.4833 ± 0.012 | higher |
| 1-step NLL | 14.12 ± 1.4 | 14.64 ± 1.5 | 14.08 ± 1.4 | 13.68 ± 1.5 | 13.9 ± 1.5 | lower |
| NLL @ memory-critical | 25.3 ± 1.7 | 16.73 ± 0.23 | 25.55 ± 1.5 | 15.61 ± 0.69 | 25.49 ± 1.8 | lower |
| NLL @ long eval length | -- | -- | -- | -- | -- | lower |
| Open-loop MSE at step 16 | 0.03065 ± 0.0043 | 44.59 ± 0.33 | 0.02837 ± 0.01 | 0.5055 ± 0.17 | 0.06053 ± 0.019 | lower |
| Open-loop MSE at step 24 | -- | -- | -- | -- | -- | lower |
| Drift slope | -0.002043 ± 0.019 | 0.1259 ± 0.0023 | -0.004878 ± 0.015 | -0.07033 ± 0.013 | 0.00609 ± 0.016 | lower |
| Probe AUC (state) | 0.8903 ± 0.013 | 0.9694 ± 0.012 | 0.7028 ± 0.016 | 0.8986 ± 0.03 | 0.6986 ± 0.0069 | higher |
| Probe AUC (obs control) | 0.5514 ± 0.019 | 0.5514 ± 0.019 | 0.5514 ± 0.019 | 0.5514 ± 0.019 | 0.5514 ± 0.019 | higher |
| Effective horizon | 20 ± 2.8 | 32 ± 0 | 4 ± 0 | 32 ± 0 | 4 ± 0 | higher |
| Return (real env) | -0.125 ± 0.088 | -- | 0.0625 ± 0.13 | 1 | -- | higher |
| Decision accuracy (agent) | 0.4375 ± 0.044 | -- | 0.5312 ± 0.066 | 1 | -- | higher |
| Forgetting on T0 (acc) | -- | -- | -- | -- | -- | lower |
| Backbone params | 2.243e+05 ± 0 | 3.752e+05 ± 0 | 2.99e+05 ± 0 | 3.007e+05 ± 0 | 3.752e+05 ± 0 | lower |
| Step latency (ms) | 0.6597 ± 0.11 | 0.9395 ± 0.0097 | 0.4621 ± 0.0058 | 4.874 ± 0.27 | 0.7077 ± 7.1e-05 | lower |
| Train pass (ms) | 7.227 ± 0.91 | 4.243 ± 0.0064 | 2.762 ± 0.039 | 827.4 ± 1.6 | 3.767 ± 0.014 | lower |

_Seeds per model: gru=2, hope-attention=2, lstm=2, supreme=2, supreme-cms=2._


## Open-loop imagination error by horizon

| mse_at_horizon | gru | hope-attention | lstm | supreme | supreme-cms |
|---|---|---|---|---|---|
| 1 | 2.198 | 5.334 | 2.188 | 2.387 | 2.194 |
| 2 | 2.279 | 8.334 | 2.273 | 2.585 | 2.293 |
| 4 | 2.342 | 6.792 | 2.336 | 2.495 | 2.34 |
| 8 | 2.721 | 7.116 | 2.71 | 2.867 | 2.713 |
| 16 | 0.03065 | 44.59 | 0.02837 | 0.5055 | 0.06053 |
| 24 | -- | -- | -- | -- | -- |


## Cue retention by delay (linear probe accuracy)

`state` probes the recurrent features h_t; `obs` probes the latent z_t alone and is the control -- once the cue leaves the screen it falls to chance, so any gap above it is memory.

| delay | gru (state) | gru (obs) | hope-attention (state) | hope-attention (obs) | lstm (state) | lstm (obs) | supreme (state) | supreme (obs) | supreme-cms (state) | supreme-cms (obs) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 |
| 2 | 1.000 | 0.578 | 1.000 | 0.578 | 1.000 | 0.578 | 1.000 | 0.578 | 1.000 | 0.578 |
| 4 | 1.000 | 0.478 | 1.000 | 0.478 | 0.956 | 0.478 | 0.533 | 0.478 | 0.867 | 0.478 |
| 8 | 1.000 | 0.489 | 0.989 | 0.489 | 0.433 | 0.489 | 0.967 | 0.489 | 0.389 | 0.489 |
| 16 | 0.900 | 0.478 | 0.944 | 0.478 | 0.400 | 0.478 | 0.822 | 0.478 | 0.456 | 0.478 |
| 24 | 0.689 | 0.444 | 0.956 | 0.444 | 0.422 | 0.444 | 0.889 | 0.444 | 0.467 | 0.444 |
| 32 | 0.533 | 0.444 | 0.867 | 0.444 | 0.411 | 0.444 | 0.978 | 0.444 | 0.411 | 0.444 |


## Cost

Measured on the machine named in each run's `meta.device`.
