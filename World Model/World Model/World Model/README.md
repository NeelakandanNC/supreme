# Model 1 — World Model (baseline)

The control condition: **Ha & Schmidhuber, *World Models* (2018)**, arXiv:1803.10122.

```
V   ConvVAE, 32-d latent, 4,348,547 params   ->  ../wmcore/vision/conv_vae.py   SHARED
M   LSTM(256) + MDN head                     ->  lstm_backbone.py  +  SHARED head
C   linear controller, CMA-ES                ->  ../wmcore/controller/          SHARED
```

**Only `lstm_backbone.py` lives here**, and that is the whole point. It contains
the recurrent core of the MDN-RNN — the single component that `../Supreme/`
replaces. Everything else in the pipeline is shared code in `../wmcore/`, so the
two models cannot differ anywhere else even by accident.

The MDN half of "MDN-RNN" is *not* here: it is in `wmcore/memory/base.py`,
shared with every other backbone. If the baseline had its own output head, a
difference in likelihood would confound what the model remembers with how it
expresses uncertainty, and no ablation could separate them. See
`../docs/design.md` §2.

## Registered backbones

| key | what it is | backbone params (input 34, hidden 256) |
|---|---|---|
| `lstm` | the paper's core: 1-layer LSTM, 256 units | 299,008 |
| `gru` | a GRU control, so "the baseline is just a weak RNN" can be tested | 224,256 |

`lstm` at these sizes reproduces the paper's reported counts: V is 4,348,547
exactly, and the full M model (core + shared MDN head) is 422,882.

## Run it

```bash
cd ..                                    # project root
python wm.py study -c configs/tmaze.yaml --backbones lstm
```

Results land in `runs/lstm/seed0/`. To compare against Supreme, name both:

```bash
python wm.py study -c configs/tmaze.yaml --backbones lstm,supreme
```
