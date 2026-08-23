"""C -- the controller.

Ha & Schmidhuber deliberately make C as small as possible: a single linear map
from ``[z_t ; h_t]`` to an action, ~870 parameters on CarRacing.  The argument
is that if the controller is trivially simple, then whatever competence the
agent shows must live in V and M.  For this study that argument is doing even
more work than in the original paper: because C is linear and identical in both
legs, a difference in return is a difference in how much task-relevant
information the memory state exposes.

Because C is tiny it is trained with an evolution strategy rather than
backpropagation -- no gradients need to flow through the environment, and the
search is trivially parallel across the M4's cores.
"""
from __future__ import annotations

import numpy as np

from wmcore.envs.spaces import Box, Discrete


class LinearController:
    """``action = f(W [z ; h] + b)``, held as a flat numpy parameter vector.

    Kept in numpy rather than torch on purpose: CMA-ES evaluates dozens of
    candidates per generation in worker processes, and a numpy matmul on a
    (867,) parameter vector avoids torch's per-call overhead entirely.
    """

    def __init__(self, input_dim: int, action_space: Discrete | Box):
        self.input_dim = int(input_dim)
        self.action_space = action_space
        self.output_dim = (action_space.n if isinstance(action_space, Discrete)
                           else int(np.prod(action_space.shape)))
        self.W = np.zeros((self.output_dim, self.input_dim), dtype=np.float64)
        self.b = np.zeros(self.output_dim, dtype=np.float64)

    # ------------------------------------------------------- parameters --
    @property
    def n_params(self) -> int:
        return self.output_dim * (self.input_dim + 1)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b])

    def set_params(self, params: np.ndarray) -> "LinearController":
        params = np.asarray(params, dtype=np.float64).ravel()
        if params.size != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {params.size}")
        split = self.output_dim * self.input_dim
        self.W = params[:split].reshape(self.output_dim, self.input_dim)
        self.b = params[split:]
        return self

    # ------------------------------------------------------------ act ----
    def act(self, features: np.ndarray):
        """``features`` is [z_t ; h_t] (or just z_t in the no-memory ablation)."""
        logits = self.W @ np.asarray(features, dtype=np.float64) + self.b
        if isinstance(self.action_space, Discrete):
            return int(np.argmax(logits))
        # CarRacing: steer in [-1, 1], gas and brake in [0, 1].  tanh then
        # rescale to the box, as in the World Models reference implementation.
        low = np.asarray(self.action_space.low, dtype=np.float64).ravel()
        high = np.asarray(self.action_space.high, dtype=np.float64).ravel()
        squashed = np.tanh(logits)
        return (low + (squashed + 1.0) * 0.5 * (high - low)).astype(np.float32)
