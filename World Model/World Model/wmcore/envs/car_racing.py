"""CarRacing adapter -- the reference environment from Ha & Schmidhuber (2018).

Included so the study has a recognisable, externally comparable data point
alongside the synthetic memory suite.  It is *not* the primary benchmark:
CarRacing is close to fully observable from a single frame, so it cannot
discriminate between memory layers, and its CMA-ES variance is large enough to
hide any difference that does exist.  Reviewers will still want to see it.

Cost note for a 16 GB M4: CarRacing steps at roughly 200-400 Hz headless per
process.  With the defaults (400 rollouts x 300 steps, action_repeat 2) data
collection takes about 15-25 minutes across 6 workers, and the resulting frame
store is ~1.5 GB on disk.
"""
from __future__ import annotations

import numpy as np

from wmcore.envs.base import Env
from wmcore.envs.spaces import Box, Discrete


class GymnasiumAdapter(Env):
    """Wrap a gymnasium env into this repo's :class:`~wmcore.envs.base.Env`."""

    def __init__(self, gym_env):
        self._env = gym_env
        space = gym_env.action_space
        if space.__class__.__name__ == "Discrete":
            self.action_space = Discrete(int(space.n))
        else:
            self.action_space = Box(
                low=np.asarray(space.low, dtype=np.float32),
                high=np.asarray(space.high, dtype=np.float32),
            )
        obs_space = gym_env.observation_space
        self.observation_shape = tuple(obs_space.shape)  # type: ignore[arg-type]

    def reset(self, *, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.uint8), dict(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.uint8), float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:
        self._env.close()


def make_car_racing(
    *,
    image_size: int = 64,
    action_repeat: int = 2,
    max_episode_steps: int = 300,
    domain_randomize: bool = False,
    continuous: bool = True,
) -> Env:
    """Build the CarRacing env used in this study.

    Raises a helpful error if gymnasium/Box2D are missing, since Box2D wheels on
    Apple silicon are the single most common install failure for this repo.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CarRacing needs gymnasium with the Box2D extra:\n"
            "    pip install 'gymnasium[box2d]' swig\n"
            "On Apple silicon install swig first -- box2d-py builds from source."
        ) from exc

    from wmcore.envs.wrappers import ActionRepeat, ResizeObservation, TimeLimit

    env_id = "CarRacing-v3"
    try:
        raw = gym.make(env_id, render_mode="rgb_array",
                       domain_randomize=domain_randomize, continuous=continuous)
    except Exception:  # older gymnasium
        raw = gym.make("CarRacing-v2", render_mode="rgb_array",
                       domain_randomize=domain_randomize, continuous=continuous)

    env: Env = GymnasiumAdapter(raw)
    env = ActionRepeat(env, action_repeat)
    env = ResizeObservation(env, image_size)
    env = TimeLimit(env, max_episode_steps)
    return env
