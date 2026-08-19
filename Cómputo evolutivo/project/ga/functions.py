"""Named objective functions f(x). The GA maximizes or minimizes these."""

from __future__ import annotations

import math
from typing import Callable

Vector = tuple[float, ...]
Objective = Callable[[Vector], float]


def x_squared(xs: Vector) -> float:
    """Goldberg lecture: f(x) = x²."""
    return xs[0] ** 2


def shifted_parabola(xs: Vector) -> float:
    """Smooth bowl with minimum 0 at x = 3."""
    return (xs[0] - 3.0) ** 2


def sphere(xs: Vector) -> float:
    """Σ x_i². Minimum 0 at the origin."""
    return sum(v * v for v in xs)


def himmelblau(xs: Vector) -> float:
    """Classic 2-D test; four minima with f ≈ 0."""
    x, y = xs[0], xs[1]
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def sine_peak(xs: Vector) -> float:
    """Multimodal 1-D: x · sin(x). Maximize on ~[0, 4π]."""
    x = xs[0]
    return x * math.sin(x)


FUNCTIONS: dict[str, Objective] = {
    "x_squared": x_squared,
    "shifted_parabola": shifted_parabola,
    "sphere": sphere,
    "himmelblau": himmelblau,
    "sine_peak": sine_peak,
}


def get_function(name: str) -> Objective:
    if name not in FUNCTIONS:
        known = ", ".join(sorted(FUNCTIONS))
        raise ValueError(f"unknown function {name!r}. Choose one of: {known}")
    return FUNCTIONS[name]
