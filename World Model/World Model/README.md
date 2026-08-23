# World Model — laptop-scale reproduction, built for an apples-to-apples memory ablation

A clean-room implementation of **World Models** (Ha & Schmidhuber, 2018) —
V (vision) + M (memory) + C (controller) — sized to train and benchmark end to
end on an **Apple M4 MacBook with 16 GB of unified memory** (~10 GB usable).

It is built for one specific experiment: **swap M's recurrent core and change
nothing else.**

| folder | model | its memory core |
|---|---|---|
| [`World Model/`](World%20Model/) | Ha & Schmidhuber (2018) — the control | LSTM(256), the MDN-RNN core |
| [`Supreme/`](Supreme/) | this work | Hope: self-modifying Titans → Continuum Memory System |

Those two folders are the *entire* difference between the two models. V, C, the
environments, the data pipeline, the MDN output head, the loss, the optimiser
and the whole benchmark live once, in `wmcore/`, and are used verbatim by both.

---

## Quick start

```bash
cd "World Model"
./setup_env.sh                 # creates .venv on Python 3.12 and installs deps
source .venv/bin/activate
python wm.py info              # device, registered envs, registered memory layers
make test                      # correctness tests (23) -- run before trusting any number
make smoke                     # full pipeline on a tiny config, ~2 minutes
```

Then the real thing:

```bash
python wm.py study -c configs/tmaze.yaml --backbones lstm
python wm.py report runs
```

The full A/B with ablations is one flag:

```bash
python wm.py study -c configs/tmaze.yaml \
    --backbones lstm,supreme,supreme-titans,supreme-cms
```

| backbone | what it is |
|---|---|
| `lstm` | the MDN-RNN core (Ha & Schmidhuber 2018) — the control |
| `supreme` | Hope: self-modifying Titans → Continuum Memory System |
| `supreme-titans` | Hope with CMS removed |
| `supreme-cms` | the baseline LSTM with CMS bolted on |

---

## Layout

```
World Model/
├── wm.py                 CLI shim  (python wm.py <command>)
├── configs/              experiment configs (YAML, with inheritance + CLI overrides)
│
├── World Model/          ◀── MODEL 1: the baseline (Ha & Schmidhuber 2018)
│   └── lstm_backbone.py      the MDN-RNN's recurrent core. That is all.
├── Supreme/              ◀── MODEL 2: this work
│   ├── titans.py             self-modifying Titans   (paper §8.1-8.2)
│   ├── cms.py                Continuum Memory System (paper §7.1)
│   └── hope.py               their composition       (paper §8.3)
│
├── wmcore/               SHARED CORE — used verbatim by BOTH models
│   ├── config.py         typed config + assert_comparable()
│   ├── envs/             CarRacing adapter + purpose-built memory environments
│   ├── data/             rollout collection, memmapped store, torch datasets
│   ├── vision/           V: ConvVAE (frozen, shared)
│   ├── memory/           M: backbone contract + shared MDN head + registry
│   ├── controller/       C: linear controller, CMA-ES / OpenAI-ES, agent rollout
│   ├── dream/            latent-space imagination environment
│   ├── train/            the four training stages
│   └── bench/            the memory benchmark suite + reporting
├── tests/                correctness tests (`make test`)
├── scripts/
│   └── sweep_horizon.py  the main figure: quality vs. required memory horizon
└── docs/
    ├── design.md         how the comparison is kept honest
    ├── benchmarks.md     what every metric means
    └── findings.md       measured baseline results, and what they imply
```

**Why two small folders instead of two projects.** The obvious layout is a
self-contained copy of the pipeline per model — and it cannot support the claim
this study makes. With two copies of the data loader, the VAE, the loss and the
controller, the arms drift, and every reported difference becomes "the memory
layer, plus whatever else diverged"; no diff of two 60-file trees will tell you
which. With one shared core there is exactly one axis of variation, and it is the
axis the paper is about. A model folder *cannot* change the training loop,
because it contains no training loop. See [docs/design.md](docs/design.md).

---

## The pipeline

| Stage | Command | What it produces | Cost on M4 (`configs/tmaze.yaml`) |
|---|---|---|---|
| 1. Collect | `wm.py collect` | memmapped frame store (~680 MB) | ~1 min, 6 workers |
| 2. Train V | `wm.py vae` | `vae_*.pt`, frozen and shared | ~3 min |
| 3. Encode | `wm.py latents` | `latents.npz` (~30 MB) | ~1 min |
| 4. Train M | `wm.py memory` | `memory.pt` | ~4 min |
| 5. Train C | `wm.py controller` | `controller.npz` | ~12 min |
| 6. Benchmark | `wm.py bench` | `bench.json` | ~2 min |
| — | `wm.py study` | all of the above, for every backbone | — |
| — | `wm.py report runs` | `REPORT.md` + figures | seconds |

Stage 3 is what makes the study affordable: after V is frozen, M trains on a
30 MB array of 32-d vectors instead of a 1.5 GB frame store, so many seeds and
many sequence lengths cost minutes rather than hours.

---

## Environments

| id | What it tests | Why it is here |
|---|---|---|
| `TMaze-v0` | retention of one cue across a tunable distractor corridor | the memory-horizon axis — the study's independent variable |
| `SequenceRecall-v0` | holding *k* associations at once and retrieving one on demand | capacity, as opposed to retention |
| `CarRacing-v0` | the reference task from the original paper | external comparability |

The two synthetic environments are pure numpy (no gymnasium, no Box2D, no
pygame), render at 64×64 like the paper, and emit ground-truth memory variables
in `info` so the benchmark's probes are exact rather than inferred. Critically,
they place the memory requirement **inside the world-model loss**: the frame
after the decision reveals the answer, so that one transition is unpredictable
unless the state still holds the cue. The environment tags it `memory_critical`
and the benchmark reports NLL on exactly those steps.

CarRacing is included for comparability, not as the discriminating benchmark —
it is close to fully observable from a single frame, so a memory layer has little
to do there.

---

## What gets measured

Full definitions in [docs/benchmarks.md](docs/benchmarks.md); measured baseline
numbers in [docs/findings.md](docs/findings.md).

0. **Decision accuracy** (the headline) — at the one transition per episode that
   needs memory, does M predict the frame that actually comes next? Chance is
   exactly 0.5 and hedging scores at chance, so it cannot be gamed the way NLL can.
1. **Teacher-forced NLL**, overall and **restricted to memory-critical steps**
2. **Length extrapolation** — evaluate on windows longer than training
3. **Open-loop imagination** — divergence vs horizon, and its slope
4. **Retention probes** — linear decodability of the cue from `h_t` vs delay,
   with the observation-only control that says what is memory and what is just
   visible on screen
5. **Continual learning** — task-incremental forgetting (BWT/FWT)
6. **Control** — real-environment return, plus the dream-trained-controller
   transfer test
7. **Cost** — parameters, latency, throughput, peak RSS

---

## Adding a memory layer

Two steps, and nothing else in the repository changes:

Drop a folder next to `World Model/` and `Supreme/`; it is discovered
automatically.

```python
# Your Model/backbone.py   (+ an __init__.py that imports it)
from wmcore.memory.base import MemoryBackbone
from wmcore.memory.registry import register_backbone

class YourBackbone(MemoryBackbone):
    name = "yours"

    def initial_state(self, batch_size, device): ...
    def forward(self, u, state=None):  ...   # (B, L, input_dim) -> (B, L, feature_dim)
    def step(self, u, state):          ...   # (B, input_dim)    -> (B, feature_dim)
    def state_features(self, state):   ...   # (B, feature_dim), for C and for probes

register_backbone("yours", YourBackbone)
```

```bash
python wm.py study -c configs/tmaze.yaml --backbones lstm,yours
```

Constraints to respect so the comparison stays valid: `feature_dim` must equal
`memory.hidden_dim`, and the backbone's parameter count should be within ~10% of
the LSTM's (~300 k at `hidden_dim=256`). `wm.py study` machine-checks the rest.

---

## Read this before running the comparison

Two things were measured on the baseline that change how the study must be run.
Both are documented with numbers in [docs/findings.md](docs/findings.md).

**1. The plain world-model objective barely rewards remembering.** In a
delayed-recall episode exactly one transition needs memory, it carries ~1/L of
the loss, and getting it right saves ~0.7 nats out of ~25. At the default weight
a converged LSTM predicts the *midpoint* of the two possible successors and
scores at chance — while a probe shows the cue was in its state earlier in the
corridor. The model is optimising correctly; the objective just does not care.

Consequence: the memory configs ship `memory.critical_loss_weight: 20`, applied
identically to both models. Without it every memory layer scores at chance and
nothing can be distinguished. With it, decision accuracy goes 0.5 → 1.0 *and*
unweighted NLL improves, so it is a fix, not a trade-off. Report both settings.

**2. Average NLL does not distinguish memory layers.** LSTM and GRU tie at
12.87 vs 12.88 one-step NLL on the same data while their effective memory
horizons differ by 3x (4 steps vs 12). Lead with decision accuracy and the
retention curve; average NLL belongs in the appendix.

**3. Decision accuracy is close to a step function** — a model either learns the
cue→outcome rule or it does not, so per-seed values cluster near 0.5 or near 1.0.
Report the mean over ≥3 seeds *and* the number of seeds that learned the rule,
and make sure the training budget is large enough that the baseline is not
merely undertrained.

## Reproducibility

* Every run writes `config.json`, `metrics.jsonl` and `bench.json`; the report is
  regenerated from artefacts alone, no retraining.
* `make sweep SEEDS="0 1 2"` runs the full A/B per seed; the report aggregates
  mean ± s.e.m.
* `assert_comparable()` aborts a study whose two legs differ anywhere outside the
  memory backbone.
* `make test` runs 23 correctness tests. The load-bearing one is
  `test_backbone_forward_matches_step`: a backbone whose sequence pass and
  single-step pass disagree trains fine, benchmarks fine on teacher-forced
  metrics, and is silently wrong in every open-loop, dream and control result.
  Any new memory layer must pass it.

## References

* D. Ha, J. Schmidhuber. *World Models.* arXiv:1803.10122, 2018.
* A. Behrouz, M. Razaviyayn, P. Zhong, V. Mirrokni. *Nested Learning: The
  Illusion of Deep Learning Architecture.* NeurIPS 2025, arXiv:2512.24695.
