"""Typed experiment configuration.

Design rules
------------
* One config object describes an *entire* study leg (V, M, C, data, bench).
* The two models under comparison share a single base config and differ only
  in ``memory.backbone`` and its ``memory.backbone_kwargs``.  If a diff between
  two run configs touches anything else, the comparison is not apples-to-apples
  and :func:`assert_comparable` will say so.
* Defaults are chosen for a 16 GB M4 MacBook (~10 GB usable).  Every default
  below has been sized so a full baseline study fits in that budget.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, get_type_hints


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Which environment to learn a world model of."""

    id: str = "TMaze-v0"
    """Registry key -- see ``wmcore.envs.registry``."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Environment constructor kwargs (corridor length, distractors, ...)."""

    image_size: int = 64
    """All observations are resized to ``image_size x image_size x 3`` uint8,
    matching Ha & Schmidhuber (2018)."""

    max_episode_steps: int = 300
    action_repeat: int = 1


@dataclass
class DataConfig:
    """Rollout collection and on-disk dataset layout."""

    root: str = "runs/data"
    n_rollouts: int = 400
    """Ha & Schmidhuber use 10,000 rollouts.  That is ~123 GB of 64x64 frames
    and days of CPU time.  400 x 300 steps = 120k frames ~= 1.5 GB, which is
    what a laptop study can actually afford.  Scale up with --data.n_rollouts
    if you have the disk."""

    steps_per_rollout: int = 300
    shard_size: int = 50
    """Rollouts per .npz shard.  Small shards keep peak RAM low because the
    frame dataset memory-maps one shard at a time."""

    policy: str = "random"
    """``random`` (uniform / Brownian) or ``mixed`` (random + partially trained
    controller), used to widen state coverage for the second data round."""

    val_fraction: float = 0.1
    frames_in_ram: int = 40_000
    """Upper bound on frames held in RAM by the VAE dataset (uint8, 64x64x3 =
    12 KB each, so 40k ~= 0.5 GB)."""

    sequence_length: int = 128
    """Training BPTT length for the memory model."""

    eval_sequence_length: int = 512
    """Longer than training on purpose: length extrapolation is one of the
    memory metrics."""


@dataclass
class VisionConfig:
    """V: the convolutional VAE.  Identical for both models."""

    latent_dim: int = 32
    channels: tuple[int, ...] = (32, 64, 128, 256)
    beta: float = 1.0
    free_bits: float = 0.5
    """Per-dimension KL floor (nats).  Ha & Schmidhuber clamp the KL term for
    the VizDoom experiment; free-bits is the modern, better-behaved form of the
    same idea and stops posterior collapse on the low-entropy memory envs."""

    recon_loss: str = "l2_sum"

    checkpoint: str | None = None
    """Explicit path to a pre-trained V.  Set by the continual-learning runner so
    that every task variant is encoded by the *same* frozen vision model -- the
    variants differ only in the cue->action mapping, so a per-variant V would add
    a spurious representational shift on top of the mapping shift."""

    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0


@dataclass
class MemoryConfig:
    """M: the sequence model.  **This is the only section that differs
    between the baseline and Supreme.**"""

    backbone: str = "lstm"
    """Registry key of the recurrent backbone.  ``lstm`` reproduces the
    MDN-RNN of Ha & Schmidhuber (2018); Supreme registers its own key."""

    backbone_kwargs: dict[str, Any] = field(default_factory=dict)

    hidden_dim: int = 256
    """Width of the recurrent state, and therefore of the feature the
    controller sees.  Held fixed across models so C has the same input size."""

    # ---- shared output head (identical for every backbone) ----
    n_mixtures: int = 5
    """MDN components.  The head is deliberately *shared* between models: if
    the baseline had an MDN head and Supreme had a Gaussian head, two things
    would change at once and the ablation would be uninterpretable."""

    predict_reward: bool = True
    predict_done: bool = True
    temperature: float = 1.0
    """Sampling temperature tau for dream rollouts (Ha & Schmidhuber Sec. 4.4)."""

    # ---- optimisation (identical for every backbone) ----
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    reward_loss_weight: float = 1.0
    done_loss_weight: float = 1.0

    critical_loss_weight: float = 1.0
    """Relative weight of ``memory_critical`` transitions in the training loss.

    Why this knob has to exist
    --------------------------
    A delayed-recall episode contains exactly **one** transition that cannot be
    predicted without memory (once the answer is revealed, memory is no longer
    needed).  At sequence length L that transition carries ~1/L of the loss, and
    the most the model can gain by getting it right is the entropy of the
    outcome -- about 0.69 nats for a binary cue, against a per-step NLL of ~25.

    We measured this: with weight 1.0 a converged LSTM world model predicts
    almost exactly the midpoint of the two possible successors and scores at
    chance on the decision metric, while a linear probe shows the cue *was* held
    in the state earlier in the corridor.  The model is not failing to remember;
    it is correctly optimising an objective that barely rewards remembering.

    That is a genuine finding about world-model objectives and is worth
    reporting, but with weight 1.0 the benchmark has no headroom -- every
    memory layer sits at chance and nothing can be distinguished.  Raising this
    weight restores headroom.  It is applied identically to both models, it
    changes no architecture, and it must be reported alongside the results."""

    param_match: bool = True
    """If True, training scripts log a warning when the two models' backbone
    parameter counts differ by more than ``param_match_tol``."""

    param_match_tol: float = 0.10


@dataclass
class ControllerConfig:
    """C: the linear controller trained with CMA-ES.  Identical for both."""

    hidden_input: bool = True
    """Feed [z_t ; h_t] (True, the paper's full model) or z_t only (False, the
    ablation that isolates how much the memory actually contributes)."""

    algo: str = "cma"        # "cma" | "openai_es"
    population: int = 32
    generations: int = 60
    sigma_init: float = 0.5
    episodes_per_candidate: int = 4
    eval_episodes: int = 16
    n_workers: int = 6
    """M4 has 10 cores; 6 workers leaves headroom for the OS and avoids
    memory pressure from six copies of V and M."""

    train_in_dream: bool = False
    """Train C inside M's imagination (Ha & Schmidhuber Sec. 5) instead of the
    real environment.  This is *the* strongest single test of a memory layer:
    a controller can only transfer if the dream is faithful."""

    dream_horizon: int = 300


@dataclass
class BenchConfig:
    """Memory-focused evaluation suite."""

    horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    """Open-loop imagination horizons for the divergence curve."""

    probe_delays: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 200)
    """Cue-to-probe delays for the linear-readout retention curve."""

    n_eval_sequences: int = 256
    continual_tasks: int = 4
    continual_epochs_per_task: int = 4
    latency_batch: int = 1
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"
    memory_budget_mb: float = 9500.0

    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    # ---------------------------------------------------------------- io ---
    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name / f"seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    # ------------------------------------------------------------ loading --
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        """Load YAML/JSON, then apply ``a.b=c`` dot-path overrides."""
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)

        base = data.pop("_base_", None)
        if base:
            parent = Config.load(path.parent / base).to_dict()
            data = _deep_merge(parent, data)

        cfg = cls.from_dict(data)
        for ov in overrides or []:
            cfg.apply_override(ov)
        return cfg

    def apply_override(self, expr: str) -> None:
        """Apply a single ``dotted.path=value`` override (value parsed as YAML)."""
        if "=" not in expr:
            raise ValueError(f"override must look like a.b=c, got {expr!r}")
        key, _, value = expr.partition("=")
        try:
            import yaml

            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value
        target: Any = self
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise AttributeError(f"unknown config key: {key}")
        current = getattr(target, leaf)
        if isinstance(current, tuple) and isinstance(parsed, list):
            parsed = tuple(parsed)
        setattr(target, leaf, parsed)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _build(cls, data: dict):
    """Recursively instantiate nested dataclasses from a plain dict.

    ``from __future__ import annotations`` turns every field annotation into a
    string, so ``dataclasses.fields(...).type`` is unusable for dispatch here;
    we resolve the real types with ``get_type_hints`` instead.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in fields:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        ftype = hints.get(key, fields[key].type)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        elif isinstance(value, list) and _default_is_tuple(fields[key]):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _default_is_tuple(f: dataclasses.Field) -> bool:
    if f.default is not dataclasses.MISSING:
        return isinstance(f.default, tuple)
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return isinstance(f.default_factory(), tuple)  # type: ignore[misc]
        except Exception:
            return False
    return False


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


ALLOWED_DIFF_KEYS = {
    "name",
    "memory.backbone",
    "memory.backbone_kwargs",
    "out_dir",
}


def assert_comparable(cfg_a: Config, cfg_b: Config, *, strict: bool = True) -> list[str]:
    """Verify two configs differ *only* in the memory backbone.

    This is the guard that keeps the headline claim honest.  Call it before
    reporting any A/B number; it returns the list of offending dotted keys.
    """
    diffs = _diff(cfg_a.to_dict(), cfg_b.to_dict(), prefix="")
    offenders = [d for d in diffs if d not in ALLOWED_DIFF_KEYS]
    if offenders and strict:
        raise AssertionError(
            "configs differ outside the memory backbone, so the comparison is "
            f"not apples-to-apples:\n  " + "\n  ".join(offenders)
        )
    return offenders


def _diff(a: dict, b: dict, prefix: str) -> list[str]:
    keys = set(a) | set(b)
    out: list[str] = []
    for k in sorted(keys):
        path = f"{prefix}{k}"
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_diff(va, vb, prefix=f"{path}."))
        elif va != vb:
            out.append(path)
    return out
