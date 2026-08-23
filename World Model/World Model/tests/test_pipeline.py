"""End-to-end correctness tests for the data pipeline and the benchmark harness.

Where ``test_core.py`` pins down individual components, this file pins down the
properties that only emerge once the pieces are wired together -- and that are
the easiest to get silently wrong:

* sequence windows must never straddle an episode boundary;
* ``z_next[t]`` must really be ``z[t+1]``;
* the controller must see the memory state *before* it has consumed the current
  observation;
* the open-loop benchmark must actually be open-loop.

The last one is tested with an analytic oracle rather than a trained model, so
it measures the harness and nothing else.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmcore.bench.imagination import open_loop_curve  # noqa: E402
from wmcore.config import Config  # noqa: E402
from wmcore.controller.agent import WorldModelAgent  # noqa: E402
from wmcore.controller.linear import LinearController  # noqa: E402
from wmcore.data.collect import collect  # noqa: E402
from wmcore.data.dataset import LatentSequenceDataset, episode_segments  # noqa: E402
from wmcore.envs import make_env  # noqa: E402
from wmcore.envs.spaces import Discrete  # noqa: E402
from wmcore.memory import build_memory  # noqa: E402

_TMP: list[Path] = []


def _tiny_config(tmp: Path, n_rollouts: int = 16, corridor: int = 4) -> Config:
    cfg = Config()
    cfg.name = "test"
    cfg.out_dir = str(tmp / "runs")
    cfg.data.root = str(tmp / "data")
    cfg.env.id = "TMaze-v0"
    cfg.env.kwargs = {"corridor_length": corridor, "n_cues": 2,
                      "cue_steps": 2, "feedback_steps": 2, "distractors": 1}
    cfg.env.max_episode_steps = corridor + 8
    cfg.data.n_rollouts = n_rollouts
    cfg.data.steps_per_rollout = corridor + 8
    cfg.data.val_fraction = 0.25
    cfg.data.sequence_length = 4
    cfg.vision.latent_dim = 8
    return cfg


def _store_with_fake_latents(cfg: Config):
    """Collect a tiny dataset and attach synthetic latents.

    Deliberately synthetic: these tests are about indexing and alignment, and a
    real VAE would add minutes of training plus a second source of failure.
    Latent value at (rollout n, step t) is ``n * 1000 + t``, so any misalignment
    is immediately visible as an arithmetic error rather than a small drift.
    """
    store = collect(cfg)
    spec = store.spec
    n, t, d = spec.n_rollouts, spec.steps, cfg.vision.latent_dim
    mu = (np.arange(n)[:, None, None] * 1000.0
          + np.arange(t)[None, :, None]
          + np.zeros((1, 1, d))).astype(np.float32)
    np.savez(store.latents_path, mu=mu, logvar=np.full_like(mu, -10.0))
    return store


# =====================================================================
# Dataset alignment
# =====================================================================
def test_windows_never_cross_episode_boundaries():
    tmp = Path(tempfile.mkdtemp()); _TMP.append(tmp)
    cfg = _tiny_config(tmp)
    store = _store_with_fake_latents(cfg)
    idx = np.arange(cfg.data.n_rollouts)

    segments = episode_segments(store, idx)
    spans = {seg.row: [] for seg in segments}
    for seg in segments:
        spans[seg.row].append((seg.start, seg.end))

    for seq_len in (3, 4, 7, 32):
        ds = LatentSequenceDataset(store, idx, seq_len, sample_latents=False)
        for row, start, seg_end in ds.windows:
            assert any(s <= start and seg_end == e for s, e in spans[row]), \
                f"window ({row},{start},{seg_end}) is not inside any episode"
        for i in range(len(ds)):
            item = ds[i]
            valid = int(item["mask"].sum().item())
            row, start, seg_end = ds.windows[i]
            assert start + valid <= seg_end - 1 + 1, \
                "window reads past the end of its episode"


def test_z_next_is_z_shifted_by_one():
    tmp = Path(tempfile.mkdtemp()); _TMP.append(tmp)
    cfg = _tiny_config(tmp)
    store = _store_with_fake_latents(cfg)
    ds = LatentSequenceDataset(store, np.arange(cfg.data.n_rollouts), 5,
                               sample_latents=False)
    for i in range(min(20, len(ds))):
        item = ds[i]
        valid = int(item["mask"].sum().item())
        z = item["z"][:valid, 0].numpy()
        z_next = item["z_next"][:valid, 0].numpy()
        assert np.allclose(z_next, z + 1.0), \
            f"z_next is not z shifted by one: {z} vs {z_next}"


def test_memory_critical_steps_are_reachable_in_training_windows():
    """The decision transition must appear in the sampled windows.  If striding
    drops it, `nll_memory_critical` is computed over an empty set and the whole
    headline metric silently disappears."""
    tmp = Path(tempfile.mkdtemp()); _TMP.append(tmp)
    cfg = _tiny_config(tmp, n_rollouts=16, corridor=6)
    store = _store_with_fake_latents(cfg)
    ds = LatentSequenceDataset(store, np.arange(cfg.data.n_rollouts),
                               cfg.data.sequence_length, sample_latents=False)
    total = sum(int(ds[i]["memory_critical"].sum().item()) for i in range(len(ds)))
    assert total >= cfg.data.n_rollouts, \
        f"only {total} memory-critical steps across {len(ds)} windows"


# =====================================================================
# Agent timing
# =====================================================================
class _RecordingBackbone(torch.nn.Module):
    """Stands in for a backbone and records how many inputs it has consumed."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.feature_dim = hidden_dim

    def initial_state(self, b, device):
        return torch.zeros(b, self.hidden_dim, device=device)

    def forward(self, u, state=None):
        if state is None:
            state = self.initial_state(u.shape[0], u.device)
        outs = []
        for t in range(u.shape[1]):
            state = state + 1.0        # "consumed one more input"
            outs.append(state)
        return torch.stack(outs, 1), state

    def step(self, u, state):
        state = state + 1.0
        return state, state

    def state_features(self, state):
        return state

    def extra_metrics(self):
        return {}


def test_controller_sees_state_before_current_observation():
    """C must be fed h_t, the state *before* z_t is consumed.

    If C is fed the post-consumption state it sees a vector that already encodes
    the observation it is choosing an action for -- the agent then scores better
    for the wrong reason, and the dream/real comparison becomes incoherent.
    """
    from wmcore.memory.base import MemoryModule

    latent_dim, action_dim, hidden = 4, 2, 3
    backbone = _RecordingBackbone(latent_dim + action_dim, hidden)
    memory = MemoryModule(backbone, latent_dim=latent_dim, action_dim=action_dim,
                          predict_reward=False, predict_done=False)

    class _IdentityVAE(torch.nn.Module):
        def encode(self, x):
            b = x.shape[0]
            return torch.zeros(b, latent_dim), torch.zeros(b, latent_dim)

        def reparameterize(self, mu, logvar):
            return mu

    seen: list[float] = []

    class _Recorder(LinearController):
        def act(self, features):
            seen.append(float(np.asarray(features)[latent_dim]))  # first h component
            return 0

    env = make_env("TMaze-v0", image_size=64, corridor_length=3,
                   n_cues=action_dim, cue_steps=1, feedback_steps=1)
    controller = _Recorder(latent_dim + hidden, Discrete(action_dim))
    agent = WorldModelAgent(_IdentityVAE(), memory, controller, torch.device("cpu"))
    agent.rollout(env, seed=0, max_steps=10)

    # The backbone increments its state by 1 per consumed input, so the value C
    # sees at step t must be exactly t: zero consumed at t=0, one at t=1, ...
    assert seen == [float(i) for i in range(len(seen))], \
        f"controller saw post-consumption states: {seen}"


# =====================================================================
# Open-loop benchmark
# =====================================================================
@dataclass
class _OracleOut:
    logpi: torch.Tensor
    mu: torch.Tensor
    logsigma: torch.Tensor
    features: torch.Tensor
    reward: None = None
    done_logit: None = None
    state: object = None


class _ShiftOracle:
    """A perfect world model for the sequence z_{t+1} = z_t + 1.

    Predicts from *its input only*, with no state.  Therefore:

    * if the harness is genuinely open-loop and hands back the model's own
      prediction, the rollout stays exactly on the true trajectory and the error
      is zero at every horizon;
    * if the harness feeds back a stale ground-truth latent, every prediction is
      off by a constant and the error is non-zero from step one.

    That makes this a direct test of the hand-off in `open_loop_curve`.
    """

    latent_dim = 3

    def eval(self):
        return self

    def initial_state(self, b, device):
        return torch.zeros(b, 1, device=device)

    def _predict(self, z):
        mu = (z + 1.0).unsqueeze(-1)                 # (..., D, 1)
        return _OracleOut(logpi=torch.zeros_like(mu),
                          logsigma=torch.full_like(mu, -6.0),
                          mu=mu, features=torch.zeros(*z.shape[:-1], 1))

    def __call__(self, z, a, state=None):
        out = self._predict(z)
        out.state = state
        return out

    def step(self, z, a, state):
        out = self._predict(z.unsqueeze(1))
        out.state = state
        return out

    def sample_next(self, out, temperature=None):
        return out.mu.squeeze(-1)


def test_open_loop_is_actually_open_loop():
    D, B, L, context = 3, 2, 20, 5
    z = torch.arange(L, dtype=torch.float32).view(1, L, 1).repeat(B, 1, D)
    batch = {
        "z": z,
        "z_next": z + 1.0,
        "action": torch.zeros(B, L, 1),
        "mask": torch.ones(B, L),
    }
    curve = open_loop_curve(_ShiftOracle(), [batch], torch.device("cpu"),
                            horizons=(1, 5, 10), context=context, max_batches=1)
    for h, err in curve["mse_at_horizon"].items():
        assert err < 1e-8, (
            f"a perfect model has open-loop error {err:.3e} at horizon {h}; "
            "the harness is not feeding back the model's own prediction"
        )


def test_open_loop_detects_a_drifting_model():
    """Sanity in the other direction: a biased model must show growing error."""
    class _Biased(_ShiftOracle):
        def _predict(self, z):
            out = super()._predict(z)
            out.mu = out.mu + 0.5      # constant bias, compounds under feedback
            return out

    D, B, L, context = 3, 2, 20, 5
    z = torch.arange(L, dtype=torch.float32).view(1, L, 1).repeat(B, 1, D)
    batch = {"z": z, "z_next": z + 1.0,
             "action": torch.zeros(B, L, 1), "mask": torch.ones(B, L)}
    curve = open_loop_curve(_Biased(), [batch], torch.device("cpu"),
                            horizons=(1, 10), context=context, max_batches=1)
    e1 = curve["mse_at_horizon"][1]
    e10 = curve["mse_at_horizon"][10]
    assert e10 > e1 > 0, f"error should compound: step1={e1}, step10={e10}"
    assert curve["drift_slope"] > 0


# =====================================================================
# Decision accuracy
# =====================================================================
def test_decision_accuracy_oracle_and_hedger():
    """A model that predicts the true successor must score 1.0; a model that
    hedges between the two outcomes must score at chance.

    The second half is the important one: hedging is exactly the strategy that
    minimises NLL without learning the rule, so a metric that rewards it would
    be measuring the wrong thing.
    """
    from wmcore.bench.decision import decision_accuracy

    torch.manual_seed(0)
    B, L, D, A = 8, 6, 4, 2
    t_dec = 3
    label = torch.tensor([i % 2 for i in range(B)], dtype=torch.long)

    green = torch.tensor([3.0, 3.0, 3.0, 3.0])
    red = torch.tensor([-3.0, -3.0, -3.0, -3.0])
    z_next = torch.randn(B, L, D) * 0.05
    z_next[:, t_dec] = torch.where(label[:, None].bool(), green, red)

    action = torch.zeros(B, L, A)
    action[:, :, 0] = 1.0                       # always take action 0
    correct_action = torch.full((B, L), -1, dtype=torch.long)
    correct_action[:, t_dec] = 1 - label        # label==1 <=> action 0 was correct
    correct_action[:, t_dec] = torch.where(label.bool(), torch.zeros_like(label),
                                           torch.ones_like(label))

    crit = torch.zeros(B, L)
    crit[:, t_dec] = 1.0
    batch = {"z": torch.randn(B, L, D), "z_next": z_next, "action": action,
             "mask": torch.ones(B, L), "memory_critical": crit,
             "correct_action": correct_action}

    class _Model:
        def __init__(self, mode): self.mode = mode
        def eval(self): return self
        def __call__(self, z, a):
            mu = (z_next if self.mode == "oracle"
                  else torch.zeros_like(z_next)).unsqueeze(-1)
            return _OracleOut(logpi=torch.zeros_like(mu), mu=mu,
                              logsigma=torch.zeros_like(mu),
                              features=torch.zeros(B, L, 1))

    oracle = decision_accuracy(_Model("oracle"), [batch], torch.device("cpu"))
    assert oracle["accuracy"] == 1.0, oracle
    assert oracle["n"] == B

    hedger = decision_accuracy(_Model("hedge"), [batch], torch.device("cpu"))
    assert 0.3 <= hedger["accuracy"] <= 0.7, (
        f"a hedging model scored {hedger['accuracy']}; the metric is gameable")


# =====================================================================
# Runner
# =====================================================================
def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    try:
        for name, fn in tests:
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    finally:
        for tmp in _TMP:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
