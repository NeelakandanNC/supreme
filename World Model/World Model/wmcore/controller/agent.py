"""The assembled agent: V -> M -> C, and how it is rolled out.

Timing convention (matches Ha & Schmidhuber, Figure 8)::

    obs_t --V--> z_t
    a_t = C([z_t ; h_t])          # h_t is M's state *before* seeing a_t
    h_{t+1} = M(z_t, a_t, h_t)
    obs_{t+1} = env.step(a_t)

Getting this off by one is the classic silent bug in world-model code: if C is
fed ``h_{t+1}`` it sees a state that already encodes the action it is about to
choose, and the agent scores better for the wrong reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wmcore.controller.linear import LinearController
from wmcore.envs.base import Env


@dataclass
class RolloutResult:
    total_reward: float
    steps: int
    infos: list[dict] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of decision points answered correctly (memory envs only)."""
        hits = [i for i in self.infos if i.get("memory_critical")]
        return float(np.mean([i.get("was_correct", 0.0) for i in hits])) if hits else float("nan")


class WorldModelAgent:
    """Bundles a trained V and M with a (candidate) controller."""

    def __init__(self, vae, memory, controller: LinearController,
                 device: torch.device, *, use_hidden: bool = True,
                 sample_latent: bool = False):
        self.vae = vae.eval()
        self.memory = memory.eval()
        self.controller = controller
        self.device = device
        self.use_hidden = use_hidden
        self.sample_latent = sample_latent

    # ------------------------------------------------------------ pieces --
    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        mu, logvar = self.vae.encode(x)
        if self.sample_latent:
            return self.vae.reparameterize(mu, logvar)
        return mu

    def features(self, z: torch.Tensor, state) -> np.ndarray:
        if not self.use_hidden:
            return z.squeeze(0).cpu().numpy()
        h = self.memory.state_features(state)
        return torch.cat([z.squeeze(0), h.squeeze(0)], dim=-1).cpu().numpy()

    @property
    def feature_dim(self) -> int:
        return (self.memory.controller_input_dim if self.use_hidden
                else self.memory.latent_dim)

    # ---------------------------------------------------------- rollout ---
    @torch.no_grad()
    def rollout(self, env: Env, *, seed: int | None = None, max_steps: int = 1000,
                record_frames: bool = False, record_info: bool = True) -> RolloutResult:
        obs, info = env.reset(seed=seed)
        state = self.memory.initial_state(1, self.device)
        total, steps = 0.0, 0
        infos: list[dict] = []
        frames: list[np.ndarray] = []

        for _ in range(max_steps):
            if record_frames:
                frames.append(obs.copy())
            z = self.encode(obs)
            action = self.controller.act(self.features(z, state))

            a_vec = torch.from_numpy(
                env.action_space.to_vector(action)).to(self.device).unsqueeze(0)
            out = self.memory.step(z, a_vec, state)
            state = out.state

            if record_info:
                rec = dict(info)
                if info.get("memory_critical"):
                    rec["was_correct"] = float(int(action) == info.get("correct_action", -1)) \
                        if np.isscalar(action) else float("nan")
                infos.append(rec)

            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break

        return RolloutResult(total_reward=total, steps=steps, infos=infos, frames=frames)
