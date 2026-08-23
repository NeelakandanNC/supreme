"""Synthetic, pixel-based environments whose difficulty is *pure memory*.

Motivation
----------
The claim under test is narrow: "swapping the recurrent core of the M model
changes how much the world model remembers".  Standard control benchmarks are
a poor instrument for that claim -- CarRacing is mostly reactive, so a memory
layer barely moves the score and any difference is swamped by CMA-ES variance.

These environments are built so that a *single* bit (or a small set of
associations) presented at the start of an episode is required to predict a
frame that arrives ``delay`` steps later, with ``delay`` a free parameter.  That
gives the study its main independent variable: a memory-horizon axis you can
sweep, on a laptop, in minutes.

Three properties make them useful for a world-model paper specifically:

* **The memory requirement lands inside the world-model loss, not just the
  reward.**  The frame following the decision reveals the answer, so
  ``p(z_{t+1} | z_{<=t}, a_{<=t})`` at that one step is only predictable if the
  recurrent state still holds the cue.  We tag that step ``memory_critical`` so
  the benchmark can report NLL on exactly the transitions that matter.
* **Ground-truth latent variables are emitted in ``info``.**  Linear probes read
  them to produce exact retention-versus-delay curves.
* **Distractors are visually salient but uninformative**, so a model cannot
  cheat by attending to the current frame.

Environments
------------
``TMazeMemoryEnv``
    One cue at t=0, a long distractor corridor, one binary/n-ary decision.
    The classic delayed-match-to-sample task, in pixels.
``SequenceRecallEnv``
    k glyph->colour associations are shown, then one is queried after a delay.
    Tests in-context associative recall and capacity, not just retention of a
    single bit.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wmcore.envs.base import Env
from wmcore.envs.render import (
    CUE_COLORS,
    GLYPHS,
    add_distractors,
    blank,
    draw_glyph,
    fill_rect,
)
from wmcore.envs.spaces import Discrete

# Phase codes stored alongside every frame (used by the benchmark, never by the model).
PHASE_CUE, PHASE_DELAY, PHASE_DECISION, PHASE_FEEDBACK = 0, 1, 2, 3


def _variant_permutation(n: int, variant: int) -> np.ndarray:
    """Deterministic cue -> correct-action permutation for task ``variant``.

    Continual-learning experiments present variants 0, 1, 2, ... in sequence.
    Because only the *mapping* changes and the pixel statistics do not, forgetting
    shows up as a clean drop in accuracy on earlier variants rather than as a
    distribution shift the VAE would also have to absorb.
    """
    rng = np.random.default_rng(1000 + variant)
    return rng.permutation(n) if variant > 0 else np.arange(n)


class TMazeMemoryEnv(Env):
    """Delayed match-to-sample in a corridor.

    Episode structure (total length is deterministic)::

        [ cue_steps ][ corridor_length ][ 1 ][ feedback_steps ]
          show cue      distractors     pick   reveal answer
                                        ^ memory_critical transition

    Parameters
    ----------
    corridor_length:
        The memory horizon.  Sweeping this is the main experiment.
    n_cues:
        Number of distinct cues, and therefore of actions.  2 is the classic
        T-maze; larger values raise the information the state must carry
        (log2(n_cues) bits) without lengthening the episode.
    distractors:
        Number of uninformative glyphs drawn per corridor frame.  0 makes the
        corridor visually static (easy); 6 makes it high-variance noise.
    cue_steps, feedback_steps:
        Length of the cue and feedback phases.
    variant:
        Task index for continual-learning experiments; permutes cue->action.
    image_size:
        Frame side length.  64 matches Ha & Schmidhuber (2018).
    """

    def __init__(
        self,
        corridor_length: int = 24,
        n_cues: int = 2,
        distractors: int = 3,
        cue_steps: int = 2,
        feedback_steps: int = 2,
        variant: int = 0,
        image_size: int = 64,
        reward_correct: float = 1.0,
        reward_wrong: float = -1.0,
    ) -> None:
        self.corridor_length = int(corridor_length)
        self.n_cues = int(n_cues)
        self.distractors = int(distractors)
        self.cue_steps = int(cue_steps)
        self.feedback_steps = int(feedback_steps)
        self.variant = int(variant)
        self.image_size = int(image_size)
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)

        self.action_space = Discrete(self.n_cues)
        self.observation_shape = (image_size, image_size, 3)
        self._perm = _variant_permutation(self.n_cues, self.variant)

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._cue = 0
        self._answer_correct: bool | None = None

    # -- lengths ----------------------------------------------------------
    @property
    def episode_length(self) -> int:
        return self.cue_steps + self.corridor_length + 1 + self.feedback_steps

    @property
    def _t_decision(self) -> int:
        return self.cue_steps + self.corridor_length

    # -- api --------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._cue = int(self._rng.integers(self.n_cues))
        self._answer_correct = None
        return self._render(), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        if self._t == self._t_decision:
            correct = int(self._perm[self._cue])
            self._answer_correct = int(action) == correct
            reward = self.reward_correct if self._answer_correct else self.reward_wrong

        self._t += 1
        terminated = self._t >= self.episode_length
        obs = self._render()
        return obs, reward, terminated, False, self._info()

    # -- internals --------------------------------------------------------
    def _phase(self, t: int | None = None) -> int:
        t = self._t if t is None else t
        if t < self.cue_steps:
            return PHASE_CUE
        if t < self._t_decision:
            return PHASE_DELAY
        if t == self._t_decision:
            return PHASE_DECISION
        return PHASE_FEEDBACK

    def _info(self) -> dict[str, Any]:
        return {
            "t": self._t,
            "phase": self._phase(),
            "cue": self._cue,
            "correct_action": int(self._perm[self._cue]),
            # True on the observation *from which* the next frame is only
            # predictable if the cue is still in memory.
            "memory_critical": self._t == self._t_decision,
            "delay_since_cue": max(0, self._t - (self.cue_steps - 1)),
            "variant": self.variant,
        }

    def _render(self) -> np.ndarray:
        size = self.image_size
        phase = self._phase()
        frame = blank(size, "darkgrey")
        # corridor floor: a lighter vertical band, present in every phase so the
        # scene is recognisably the same environment throughout.
        fill_rect(frame, 0, size // 4, size, size // 2, "grey")

        if phase == PHASE_CUE:
            color = CUE_COLORS[self._cue % len(CUE_COLORS)]
            fill_rect(frame, size // 4, size // 4, size // 2, size // 2, color)
        elif phase == PHASE_DELAY:
            if self.distractors:
                add_distractors(frame, self._rng, self.distractors, radius=max(3, size // 12))
            # progress marker: monotone, so the frame tells you *when* you are
            # but never *what* the cue was.
            prog = (self._t - self.cue_steps) / max(1, self.corridor_length - 1)
            y = int(prog * (size - 8))
            fill_rect(frame, y, 2, 6, 6, "white")
        elif phase == PHASE_DECISION:
            # A junction: n_cues doors along the top, all identical.
            door_w = max(4, size // (2 * self.n_cues))
            for i in range(self.n_cues):
                x = int((i + 0.5) * size / self.n_cues) - door_w // 2
                fill_rect(frame, 4, x, size // 3, door_w, "white")
        else:  # PHASE_FEEDBACK
            frame[:, :] = np.array(
                (40, 200, 90) if self._answer_correct else (220, 40, 40), dtype=np.uint8
            )
        return frame


class SequenceRecallEnv(Env):
    """Associative recall over pixel frames.

    ``n_pairs`` glyph->colour associations are shown one per frame, then after a
    delay one glyph is queried and the agent must name its colour.  Where
    :class:`TMazeMemoryEnv` measures *retention* of one item over a long delay,
    this measures *capacity*: the state has to hold ``n_pairs`` bindings at once
    and retrieve the right one on demand.  That is the regime the nested-learning
    memory is designed for, so both axes are worth reporting.

    Episode structure::

        [ n_pairs ][ delay ][ 1 ][ feedback_steps ]
          study      noise   query   reveal
                             ^ memory_critical
    """

    def __init__(
        self,
        n_pairs: int = 4,
        n_values: int = 4,
        delay: int = 12,
        distractors: int = 3,
        feedback_steps: int = 2,
        variant: int = 0,
        image_size: int = 64,
        reward_correct: float = 1.0,
        reward_wrong: float = -1.0,
    ) -> None:
        if n_pairs > len(GLYPHS):
            raise ValueError(f"n_pairs <= {len(GLYPHS)} (glyph vocabulary size)")
        if n_values > len(CUE_COLORS):
            raise ValueError(f"n_values <= {len(CUE_COLORS)}")
        self.n_pairs = int(n_pairs)
        self.n_values = int(n_values)
        self.delay = int(delay)
        self.distractors = int(distractors)
        self.feedback_steps = int(feedback_steps)
        self.variant = int(variant)
        self.image_size = int(image_size)
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)

        self.action_space = Discrete(self.n_values)
        self.observation_shape = (image_size, image_size, 3)
        self._value_perm = _variant_permutation(self.n_values, self.variant)

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._glyph_ids: np.ndarray = np.arange(self.n_pairs)
        self._values: np.ndarray = np.zeros(self.n_pairs, dtype=np.int64)
        self._query = 0
        self._answer_correct: bool | None = None

    @property
    def episode_length(self) -> int:
        return self.n_pairs + self.delay + 1 + self.feedback_steps

    @property
    def _t_query(self) -> int:
        return self.n_pairs + self.delay

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        # Distinct glyphs, values sampled with replacement.
        self._glyph_ids = self._rng.permutation(len(GLYPHS))[: self.n_pairs]
        self._values = self._rng.integers(self.n_values, size=self.n_pairs)
        self._query = int(self._rng.integers(self.n_pairs))
        self._answer_correct = None
        return self._render(), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        if self._t == self._t_query:
            self._answer_correct = int(action) == self._correct_action
            reward = self.reward_correct if self._answer_correct else self.reward_wrong
        self._t += 1
        terminated = self._t >= self.episode_length
        return self._render(), reward, terminated, False, self._info()

    @property
    def _correct_action(self) -> int:
        return int(self._value_perm[self._values[self._query]])

    def _phase(self) -> int:
        if self._t < self.n_pairs:
            return PHASE_CUE
        if self._t < self._t_query:
            return PHASE_DELAY
        if self._t == self._t_query:
            return PHASE_DECISION
        return PHASE_FEEDBACK

    def _info(self) -> dict[str, Any]:
        return {
            "t": self._t,
            "phase": self._phase(),
            "cue": self._correct_action,          # the fact probes try to decode
            "query_index": self._query,
            "correct_action": self._correct_action,
            "memory_critical": self._t == self._t_query,
            # Distance from the frame that carried the queried association.
            "delay_since_cue": max(0, self._t - self._query),
            "variant": self.variant,
        }

    def _render(self) -> np.ndarray:
        size = self.image_size
        phase = self._phase()
        frame = blank(size, "black")
        half = size // 2

        if phase == PHASE_CUE:
            i = self._t
            fill_rect(frame, 0, 0, size, half, "darkgrey")
            draw_glyph(frame, GLYPHS[self._glyph_ids[i]], size // 2, half // 2,
                       size // 6, "white")
            fill_rect(frame, 0, half, size, half,
                      CUE_COLORS[int(self._values[i]) % len(CUE_COLORS)])
        elif phase == PHASE_DELAY:
            fill_rect(frame, 0, 0, size, size, "darkgrey")
            if self.distractors:
                add_distractors(frame, self._rng, self.distractors, radius=max(3, size // 12))
        elif phase == PHASE_DECISION:
            # Same layout as study frames but with the value half blanked out:
            # the model must fill it in from memory.
            fill_rect(frame, 0, 0, size, half, "darkgrey")
            draw_glyph(frame, GLYPHS[self._glyph_ids[self._query]], size // 2, half // 2,
                       size // 6, "white")
            fill_rect(frame, 0, half, size, half, "grey")
        else:  # PHASE_FEEDBACK
            frame[:, :] = np.array(
                (40, 200, 90) if self._answer_correct else (220, 40, 40), dtype=np.uint8
            )
        return frame
