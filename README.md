# Supreme — isolating the memory layer of a world model

An apples-to-apples comparison of the **MDN-RNN** memory from Ha & Schmidhuber's
*World Models* against **Hope**, the nested-learning memory of Behrouz et al.
(NeurIPS 2025) — with every other component held byte-identical, and the whole
study running on a single 16 GB Apple M4 laptop.

**Start here:** [`supreme.tex`](supreme.tex) is the write-up (needs
[`results_tables.tex`](results_tables.tex) beside it to build). The code is under
[`World Model/World Model/`](World%20Model/World%20Model/).

## The result

`TMaze-v0`, corridor 32 (a 32-step required memory horizon), 2 seeds, matched
parameters, identical everything except the memory backbone:

| | `lstm` (MDN-RNN) | `gru` | **`supreme`** (Hope) | `supreme-cms` | `hope-attention` |
|---|---|---|---|---|---|
| Decision accuracy (chance 0.5) | 0.475 ± 0.047 | 0.483 ± 0.006 | **1.000 ± 0.000** | 0.483 ± 0.017 | **1.000 ± 0.000** |
| NLL @ memory-critical | 25.55 ± 2.15 | 25.30 ± 2.34 | **15.61 ± 0.97** | 25.49 ± 2.58 | 16.73 ± 0.33 |
| Effective memory horizon | 4 | 20 ± 4 | **32** | 4 | **32** |
| Probe accuracy at delay 32 | 0.411 ± 0.078 | 0.533 ± 0.067 | **0.978 ± 0.000** | — | — |
| Real-env return | 0.06 ± 0.19 | −0.13 ± 0.13 | **1.00** (n=1) | — | — |
| Backbone params | 299,008 | 224,256 | 300,696 | 375,184 | 375,234 |

Three things to take from it, in order of how much they should shape the paper:

1. **Hope solves a horizon neither recurrent core reaches**, and a linear probe
   locates why: Hope's state still encodes the cue at delay 32, the GRU's has
   decayed to chance, the LSTM's is gone by delay 8.
2. **The Continuum Memory System contributes nothing here.** `supreme-cms` is the
   baseline LSTM plus CMS — 25% more parameters — and it lands exactly on the
   baseline across every metric. The advantage is entirely in the sequence mixer.
3. **Self-modification is not uniquely necessary for retention.**
   `hope-attention` (softmax attention in place of Titans) also reaches 1.000.
   What separates them is *imagination*: attention's open-loop rollout diverges
   (MSE 44.6 at horizon 16 vs Hope's 0.51) while Hope stays on trajectory at a
   fixed-size state. For a world model, that dissociation is the interesting
   finding.

Plus a property of the objective worth reporting on its own: maximum-likelihood
world-model training rewards retaining a task-critical cue by only ~0.7 nats out
of ~25, so at the standard weighting **every** memory layer tested scores at
chance. Quantified in `docs/findings.md` §1.

## What is and is not established

See [`World Model/World Model/docs/status.md`](World%20Model/World%20Model/docs/status.md)
for live coverage. Short version: 2 seeds not 3; one Hope seed trained 60 of 150
epochs before being interrupted; `supreme-titans` (Titans without CMS) is
implemented but unrun; SequenceRecall, the continual protocol and the corridor
sweep are outstanding. None of that is hidden in the write-up.

## Reproducing

```bash
cd "World Model/World Model"
./setup_env.sh && source .venv/bin/activate
make test                                        # 25 correctness tests
python wm.py study -c configs/tmaze_hard.yaml --backbones lstm,gru,supreme
python wm.py report runs
```

Checkpoints, the frame store and the two source PDFs are gitignored — the first
two are regenerable, and the papers are third-party copyrighted works
(arXiv:1803.10122, arXiv:2512.24695) that a public repository should not
redistribute. Everything needed to rerun and to check the numbers
(`bench.json`, `metrics.jsonl`, `config.json`, reports, figures) is committed.

## Layout

```
supreme.tex             the write-up
results_tables.tex      tables, generated from bench.json — never hand-edited
World Model/World Model/
├── World Model/        MODEL 1: the MDN-RNN core (+ GRU and attention controls)
├── Supreme/            MODEL 2: Hope — titans.py, cms.py, hope.py
├── wmcore/             SHARED core, used verbatim by both models
├── configs/  docs/  tests/  scripts/
└── runs_seeds/ runs_carracing/   benchmark artefacts
```

The two model folders are the *entire* difference between the two systems. See
`docs/design.md` for why that layout is load-bearing rather than cosmetic.
