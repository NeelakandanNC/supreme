"""Correctness tests for the shared core.

These are not smoke tests.  Each one pins down a property that, if it broke
silently, would produce a plausible-looking but wrong number in the paper.

Run with::

    python -m pytest tests -q          # if pytest is installed
    python tests/test_core.py          # standalone, no dependencies beyond torch

The single most important test here is
:func:`test_backbone_forward_matches_step`.  Every memory backbone must satisfy
it -- including Supreme.  A backbone whose sequence pass and single-step pass
disagree will train fine, benchmark fine on teacher-forced metrics, and be
quietly wrong in every open-loop, dream and control result.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmcore.config import Config, assert_comparable          # noqa: E402
from wmcore.envs import SequenceRecallEnv, TMazeMemoryEnv, make_env  # noqa: E402
from wmcore.memory import build_backbone, build_memory, mdn_mean, mdn_nll, mdn_sample  # noqa: E402
from wmcore.memory.base import MemoryBackbone  # noqa: E402
from wmcore.bench.probes import ridge_probe  # noqa: E402

ALL_BACKBONES = ("lstm", "gru", "attention", "supreme", "supreme-titans",
                 "supreme-cms", "hope-attention")


# =====================================================================
# Memory backbone contract
# =====================================================================
def test_backbone_forward_matches_step():
    """`forward(sequence)` must equal `step()` applied one timestep at a time.

    This is the contract that makes teacher-forced training and open-loop
    dreaming describe the same model.  Break it and the world model you
    evaluate is not the world model you trained.
    """
    torch.manual_seed(0)
    B, L, D = 3, 11, 7
    for name in ALL_BACKBONES:
        bb = build_backbone(name, input_dim=D, hidden_dim=16).eval()
        u = torch.randn(B, L, D)

        with torch.no_grad():
            h_seq, _ = bb(u)
            state = bb.initial_state(B, u.device)
            h_steps = []
            for t in range(L):
                h_t, state = bb.step(u[:, t], state)
                h_steps.append(h_t)
            h_stepwise = torch.stack(h_steps, dim=1)

        err = (h_seq - h_stepwise).abs().max().item()
        assert err < 1e-5, f"{name}: forward/step mismatch, max abs diff {err:.2e}"


def test_backbone_is_causal():
    """h_t must not depend on u_{>t}.  A non-causal core would leak the future
    into the memory state and every retention curve would be meaningless."""
    torch.manual_seed(0)
    B, L, D = 2, 9, 5
    for name in ALL_BACKBONES:
        bb = build_backbone(name, input_dim=D, hidden_dim=12).eval()
        u = torch.randn(B, L, D)
        u2 = u.clone()
        u2[:, L - 1] += 10.0  # perturb only the last step
        with torch.no_grad():
            h1, _ = bb(u)
            h2, _ = bb(u2)
        assert torch.allclose(h1[:, : L - 1], h2[:, : L - 1], atol=1e-6), \
            f"{name}: h_t changed when a later input changed"


def test_state_features_width_matches_hidden_dim():
    """C's input width -- and therefore the CMA-ES search dimension -- is fixed
    by this. A backbone that returns a different width breaks comparability."""
    for name in ALL_BACKBONES:
        bb = build_backbone(name, input_dim=6, hidden_dim=24)
        state = bb.initial_state(4, torch.device("cpu"))
        feats = bb.state_features(state)
        assert feats.shape == (4, 24), f"{name}: got {tuple(feats.shape)}"
        assert bb.feature_dim == 24


def test_titans_fast_path_matches_reference():
    """Supreme's batched memory update must equal the equation-by-equation one.

    ``_advance`` updates all five memories with one batched einsum because five
    separate ones are unusably slow on MPS.  ``_advance_reference`` is the
    literal transcription of Eqs. 86-88.  A hand-optimised recurrence that
    nothing verifies is how a subtly wrong model reaches a submission, so they
    are checked against each other here.
    """
    torch.manual_seed(0)
    bb = build_backbone("supreme", input_dim=12, hidden_dim=64,
                        n_heads=4, chunk_size=3).eval()
    titans = bb.mixer
    u = torch.randn(3, 11, 64)

    with torch.no_grad():
        fast = titans.initial_state(3, u.device)
        ref = titans.initial_state(3, u.device)
        for t in range(u.shape[1]):
            y_fast, fast = titans._advance(u[:, t], fast)
            y_ref, ref = titans._advance_reference(u[:, t], ref)
            assert torch.allclose(y_fast, y_ref, atol=1e-5), \
                f"outputs diverge at t={t}: {(y_fast - y_ref).abs().max():.2e}"
            assert torch.allclose(fast.M, ref.M, atol=1e-5), \
                f"memories diverge at t={t}: {(fast.M - ref.M).abs().max():.2e}"


def test_titans_memory_stays_finite_over_a_long_sequence():
    """The recurrence must not blow up.

    eta and alpha are bounded precisely to prevent this; if a config change
    unbounds them the failure is silent during a short test and catastrophic in
    a real run, so it is pinned here at a length longer than any config uses.
    """
    torch.manual_seed(0)
    bb = build_backbone("supreme", input_dim=12, hidden_dim=64, n_heads=4).eval()
    u = torch.randn(2, 300, 12)
    with torch.no_grad():
        y, state = bb(u)
    assert torch.isfinite(y).all(), "output went non-finite"
    assert torch.isfinite(state.mixer.M).all(), "memory went non-finite"
    assert float(state.mixer.M.abs().max()) < 1e4, "memory magnitude exploded"


# =====================================================================
# MDN head
# =====================================================================
def test_mdn_nll_matches_analytic_gaussian():
    """With one mixture component the MDN NLL must equal the closed-form
    Gaussian NLL, summed over latent dimensions."""
    torch.manual_seed(0)
    B, L, D = 2, 3, 4
    mu = torch.randn(B, L, D, 1)
    logsigma = torch.randn(B, L, D, 1) * 0.3
    logpi = torch.zeros(B, L, D, 1)  # log softmax of a single component = 0
    target = torch.randn(B, L, D)

    got = mdn_nll(logpi, mu, logsigma, target)
    sigma = logsigma.exp().squeeze(-1)
    m = mu.squeeze(-1)
    expected = (0.5 * ((target - m) / sigma) ** 2 + logsigma.squeeze(-1)
                + 0.5 * math.log(2 * math.pi)).sum(-1)
    assert torch.allclose(got, expected, atol=1e-5)


def test_mdn_mean_is_mixture_expectation():
    torch.manual_seed(0)
    logpi = torch.log_softmax(torch.randn(2, 3, 4, 5), dim=-1)
    mu = torch.randn(2, 3, 4, 5)
    expected = (logpi.exp() * mu).sum(-1)
    assert torch.allclose(mdn_mean(logpi, mu), expected)


def test_mdn_sample_recovers_component_mean_at_low_variance():
    """Sampling from a near-deterministic mixture must land on that component."""
    torch.manual_seed(0)
    B, L, D, K = 4, 2, 3, 3
    logpi = torch.full((B, L, D, K), -20.0)
    logpi[..., 1] = 0.0                     # component 1 has all the mass
    logpi = torch.log_softmax(logpi, dim=-1)
    mu = torch.zeros(B, L, D, K)
    mu[..., 1] = 5.0
    logsigma = torch.full((B, L, D, K), -8.0)
    s = mdn_sample(logpi, mu, logsigma, temperature=1.0)
    assert (s - 5.0).abs().max().item() < 1e-2


# =====================================================================
# Environments
# =====================================================================
def test_tmaze_structure_and_memory_critical_step():
    env = TMazeMemoryEnv(corridor_length=7, n_cues=2, cue_steps=2, feedback_steps=2)
    obs, info = env.reset(seed=3)
    infos = [info]
    rng = np.random.default_rng(0)
    while True:
        obs, r, term, trunc, info = env.step(env.action_space.sample(rng))
        infos.append(info)
        if term or trunc:
            break

    assert len(infos) == env.episode_length + 1
    crit = [i for i in infos if i["memory_critical"]]
    assert len(crit) == 1, "exactly one memory-critical transition per episode"
    assert crit[0]["t"] == env.cue_steps + env.corridor_length

    # The cue must be gone from the screen well before the decision, otherwise
    # the task is solvable without memory at all.
    decision_t = crit[0]["t"]
    assert infos[decision_t]["delay_since_cue"] >= env.corridor_length


def test_tmaze_reward_requires_the_cue():
    """Acting on the cue always wins; ignoring it wins at chance.  If this fails
    the environment is not measuring memory."""
    env = TMazeMemoryEnv(corridor_length=5, n_cues=2)
    informed, blind = 0, 0
    for ep in range(40):
        for policy in ("informed", "blind"):
            obs, info = env.reset(seed=ep)
            total = 0.0
            while True:
                a = info["correct_action"] if policy == "informed" else 0
                obs, r, term, trunc, info = env.step(a)
                total += r
                if term or trunc:
                    break
            if policy == "informed":
                informed += total
            else:
                blind += total
    assert informed == 40, "an agent that knows the cue must always be right"
    assert blind < 40 * 0.75, "a cue-blind agent must not solve the task"


def test_env_is_deterministic_given_seed():
    for build in (lambda: TMazeMemoryEnv(corridor_length=6, distractors=4),
                  lambda: SequenceRecallEnv(n_pairs=3, n_values=3, delay=5)):
        frames = []
        for _ in range(2):
            env = build()
            obs, _ = env.reset(seed=11)
            seq = [obs.copy()]
            for _ in range(5):
                obs, *_ = env.step(0)
                seq.append(obs.copy())
            frames.append(np.stack(seq))
        assert np.array_equal(frames[0], frames[1]), "same seed must give same pixels"


def test_sequence_recall_answer_is_the_queried_value():
    env = SequenceRecallEnv(n_pairs=4, n_values=4, delay=3)
    for ep in range(20):
        obs, info = env.reset(seed=ep)
        while not info["memory_critical"]:
            obs, r, term, trunc, info = env.step(0)
        obs, r, term, trunc, _ = env.step(info["correct_action"])
        assert r > 0, "answering with correct_action must be rewarded"


def test_frames_are_uint8_and_right_shape():
    for env in (make_env("TMaze-v0", image_size=64, corridor_length=3),
                make_env("SequenceRecall-v0", image_size=64, n_pairs=2, n_values=2, delay=2)):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.uint8 and obs.shape == (64, 64, 3)


# =====================================================================
# Probes
# =====================================================================
def test_ridge_probe_detects_signal_and_rejects_noise():
    rng = np.random.default_rng(0)
    n, d = 400, 8
    y = rng.integers(0, 2, size=n)
    groups = np.arange(n) // 4                      # 4 samples per "episode"

    informative = rng.standard_normal((n, d)) + y[:, None] * 4.0
    assert ridge_probe(informative, y, groups)["accuracy"] > 0.95

    noise = rng.standard_normal((n, d))
    acc = ridge_probe(noise, y, groups)["accuracy"]
    assert acc < 0.70, f"probe found signal in pure noise: {acc}"


def test_ridge_probe_split_is_by_group():
    """Every test sample's group must be absent from training; otherwise a probe
    reports ~100% for any model because neighbouring timesteps are duplicates."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, size=n)
    groups = np.arange(n) // 10
    # Features carry the group id, not the label: a leaky split would let the
    # probe memorise group -> label; a clean split cannot.
    x = np.zeros((n, 4), dtype=np.float32)
    x[:, 0] = groups
    acc = ridge_probe(x, y, groups)["accuracy"]
    assert acc < 0.80, f"group leakage suspected: {acc}"


# =====================================================================
# Config / comparability
# =====================================================================
def test_config_nested_types_and_overrides():
    cfg = Config()
    cfg.apply_override("memory.hidden_dim=128")
    cfg.apply_override("bench.horizons=[1, 4]")
    assert cfg.memory.hidden_dim == 128
    assert cfg.bench.horizons == (1, 4)


def test_assert_comparable_rejects_off_axis_differences():
    a, b = Config(), Config()
    b.memory.backbone = "gru"
    assert_comparable(a, b)                       # the allowed axis

    b.memory.lr = 3e-4
    try:
        assert_comparable(a, b)
    except AssertionError as exc:
        assert "memory.lr" in str(exc)
    else:
        raise AssertionError("a learning-rate difference must abort the study")


def test_memory_module_rejects_wrong_input_width():
    cfg = Config().memory
    try:
        build_memory(cfg, latent_dim=32, action_dim=3)
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"valid construction failed: {exc}")

    class Bad(MemoryBackbone):
        name = "bad"

        def initial_state(self, b, d): return None
        def forward(self, u, state=None): return u, None
        def step(self, u, state): return u, None
        def state_features(self, state): return torch.zeros(1)

    from wmcore.memory.base import MemoryModule
    try:
        MemoryModule(Bad(input_dim=99, hidden_dim=8), latent_dim=32, action_dim=3)
    except ValueError:
        pass
    else:
        raise AssertionError("a backbone with the wrong input width must be rejected")


# =====================================================================
# Runner
# =====================================================================
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
