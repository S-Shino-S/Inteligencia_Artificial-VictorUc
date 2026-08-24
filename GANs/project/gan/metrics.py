"""Small statistics helpers: mean, std, and histograms."""

from __future__ import annotations

import math
from typing import List


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def histogram(xs: List[float], bins: int, low: float, high: float) -> List[int]:
    """Count values into `bins` equal buckets over [low, high]. Values outside
    the range fall into the first/last bucket."""
    counts = [0] * bins
    width = (high - low) / bins
    for x in xs:
        idx = int((x - low) / width)
        idx = 0 if idx < 0 else (bins - 1 if idx >= bins else idx)
        counts[idx] += 1
    return counts
