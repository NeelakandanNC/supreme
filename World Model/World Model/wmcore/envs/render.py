"""A tiny numpy rasteriser for the synthetic memory environments.

Why not use pygame / PIL?  Because these environments are executed millions of
times during rollout collection and CMA-ES evaluation, in six worker
processes, on a laptop.  Drawing a handful of filled primitives straight into a
uint8 array is roughly an order of magnitude cheaper than going through a
surface library, has no import cost, and is trivially deterministic.

All frames are HWC uint8 in [0, 255].
"""
from __future__ import annotations

import numpy as np

# A palette of visually well-separated colours.  The VAE has to be able to tell
# these apart from a 32-d latent, so we avoid near-duplicates.
PALETTE: dict[str, tuple[int, int, int]] = {
    "black":   (12, 12, 16),
    "white":   (240, 240, 240),
    "red":     (220, 40, 40),
    "green":   (40, 200, 90),
    "blue":    (50, 90, 230),
    "yellow":  (240, 210, 40),
    "magenta": (215, 60, 200),
    "cyan":    (40, 210, 215),
    "orange":  (240, 130, 30),
    "grey":    (110, 110, 118),
    "darkgrey": (55, 55, 62),
}

CUE_COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow", "magenta", "cyan", "orange")


def blank(size: int = 64, color: str | tuple[int, int, int] = "black") -> np.ndarray:
    rgb = PALETTE[color] if isinstance(color, str) else color
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:, :] = rgb
    return frame


def fill_rect(frame: np.ndarray, y0: int, x0: int, h: int, w: int,
              color: str | tuple[int, int, int]) -> None:
    rgb = PALETTE[color] if isinstance(color, str) else color
    size = frame.shape[0]
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(size, y0 + h), min(size, x0 + w)
    if y1 > y0 and x1 > x0:
        frame[y0:y1, x0:x1] = rgb


def fill_disc(frame: np.ndarray, cy: int, cx: int, radius: int,
              color: str | tuple[int, int, int]) -> None:
    rgb = PALETTE[color] if isinstance(color, str) else color
    size = frame.shape[0]
    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius * radius
    frame[mask] = rgb


def fill_triangle(frame: np.ndarray, cy: int, cx: int, radius: int,
                  color: str | tuple[int, int, int], up: bool = True) -> None:
    rgb = PALETTE[color] if isinstance(color, str) else color
    size = frame.shape[0]
    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    dy = (ys - cy) if up else (cy - ys)
    mask = (dy >= -radius) & (dy <= radius) & (np.abs(xs - cx) <= (radius - dy) / 2 + 1)
    frame[mask] = rgb


def fill_cross(frame: np.ndarray, cy: int, cx: int, radius: int,
               color: str | tuple[int, int, int], thickness: int = 4) -> None:
    fill_rect(frame, cy - thickness // 2, cx - radius, thickness, 2 * radius, color)
    fill_rect(frame, cy - radius, cx - thickness // 2, 2 * radius, thickness, color)


#: Glyph vocabulary used as *keys* in the associative-recall environment.
GLYPHS = ("disc", "square", "triangle_up", "triangle_down", "cross", "bar_h", "bar_v", "ring")


def draw_glyph(frame: np.ndarray, glyph: str, cy: int, cx: int, radius: int,
               color: str | tuple[int, int, int]) -> None:
    """Draw one of :data:`GLYPHS`.  Shape carries the identity, colour is free."""
    if glyph == "disc":
        fill_disc(frame, cy, cx, radius, color)
    elif glyph == "square":
        fill_rect(frame, cy - radius, cx - radius, 2 * radius, 2 * radius, color)
    elif glyph == "triangle_up":
        fill_triangle(frame, cy, cx, radius, color, up=True)
    elif glyph == "triangle_down":
        fill_triangle(frame, cy, cx, radius, color, up=False)
    elif glyph == "cross":
        fill_cross(frame, cy, cx, radius, color, thickness=max(3, radius // 2))
    elif glyph == "bar_h":
        fill_rect(frame, cy - radius // 3, cx - radius, 2 * (radius // 3), 2 * radius, color)
    elif glyph == "bar_v":
        fill_rect(frame, cy - radius, cx - radius // 3, 2 * radius, 2 * (radius // 3), color)
    elif glyph == "ring":
        fill_disc(frame, cy, cx, radius, color)
        fill_disc(frame, cy, cx, max(1, radius // 2), "black")
    else:
        raise ValueError(f"unknown glyph {glyph!r}")


def add_distractors(frame: np.ndarray, rng: np.random.Generator, n: int,
                    radius: int = 6) -> None:
    """Scatter salient but *uninformative* blobs over the frame.

    Distractors are the reason this benchmark is about memory rather than about
    perception.  Without them a model can loiter on the current frame; with
    them the current frame is high-variance noise and the only stable signal is
    what the recurrent state carried forward.
    """
    size = frame.shape[0]
    for _ in range(n):
        cy = int(rng.integers(radius, size - radius))
        cx = int(rng.integers(radius, size - radius))
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        glyph = GLYPHS[int(rng.integers(len(GLYPHS)))]
        draw_glyph(frame, glyph, cy, cx, radius, color)
