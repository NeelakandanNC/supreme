"""The dream: an environment whose dynamics *are* the memory model.

Ha & Schmidhuber's most striking result is that a controller trained entirely
inside M's imagination transfers to the real environment (their Section 5).  For
this study that experiment is not a curiosity -- it is the single most sensitive
end-to-end probe of a memory layer.  A controller can only learn a
memory-dependent policy in the dream if the dream itself remembers; and it can
only *transfer* if what the dream remembers is true of the real environment.

The dream exposes the same :class:`~wmcore.envs.base.Env` interface as a real
environment, so the identical CMA-ES code trains C in either.
"""
from __future__ import annotations

import numpy as np
import torch

from wmcore.envs.base import Env
from wmcore.envs.spaces import Box, Discrete


class DreamEnv(Env):
    """Latent-space rollout driven by M.

    Parameters
    ----------
    memory:
        A trained :class:`~wmcore.memory.base.MemoryModule`.
    initial_latents:
        (N, D) pool of real z_0 vectors to start dreams from.  Dreaming from
        noise puts the model far off the data manifold immediately; the paper
        seeds from real initial frames and so do we.
    temperature:
        Sampling temperature tau.  Ha & Schmidhuber, Section 5: raising tau
        makes the dream *harder and noisier*, which is what stops the
        controller from discovering adversarial policies that exploit the model
        instead of solving the task.  tau ~ 1.15 was their sweet spot.
    horizon:
        Steps before truncation.
    done_threshold:
        Probability above which M's termination head ends the dream.
    """

    def __init__(
        self,
        memory,
        initial_latents: np.ndarray,
        action_space: Discrete | Box,
        device: torch.device,
        *,
        temperature: float = 1.15,
        horizon: int = 300,
        done_threshold: float = 0.5,
        seed: int = 0,
    ):
        self.memory = memory.eval()
        self.initial_latents = np.asarray(initial_latents, dtype=np.float32)
        self.action_space = action_space
        self.device = device
        self.temperature = float(temperature)
        self.horizon = int(horizon)
        self.done_threshold = float(done_threshold)
        self.observation_shape = (memory.latent_dim,)  # latent, not pixels

        self._rng = np.random.default_rng(seed)
        self._state = None
        self._z: torch.Tensor | None = None
        self._t = 0

    # -- api --------------------------------------------------------------
    @torch.no_grad()
    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(len(self.initial_latents)))
        self._z = torch.from_numpy(self.initial_latents[idx]).to(self.device).unsqueeze(0)
        self._state = self.memory.initial_state(1, self.device)
        self._t = 0
        return self._z.squeeze(0).cpu().numpy(), {"t": 0, "dream": True}

    @torch.no_grad()
    def step(self, action):
        assert self._z is not None, "call reset() first"
        a = torch.from_numpy(self.action_space.to_vector(action)).to(self.device).unsqueeze(0)
        out = self.memory.step(self._z, a, self._state)
        self._state = out.state

        z_next = self.memory.sample_next(out, self.temperature).squeeze(1)  # (1, D)
        reward = float(out.reward.squeeze().item()) if out.reward is not None else 0.0
        terminated = False
        if out.done_logit is not None:
            terminated = bool(torch.sigmoid(out.done_logit).squeeze().item() > self.done_threshold)

        self._z = z_next
        self._t += 1
        truncated = self._t >= self.horizon
        return z_next.squeeze(0).cpu().numpy(), reward, terminated, truncated, \
            {"t": self._t, "dream": True}

    # -- helper -----------------------------------------------------------
    def state_features(self) -> np.ndarray:
        """[z_t ; h_t] for the controller, without leaving the dream."""
        h = self.memory.state_features(self._state)
        return torch.cat([self._z.squeeze(0), h.squeeze(0)], dim=-1).cpu().numpy()


def collect_initial_latents(store, n: int = 512) -> np.ndarray:
    """Sample real episode-start latents to seed dreams from."""
    with np.load(store.latents_path) as data:
        mu = data["mu"]
    n = min(n, mu.shape[0])
    return np.ascontiguousarray(mu[:n, 0, :])
