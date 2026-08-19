"""Entropy and information gain (ID3)."""

from __future__ import annotations

import math
from collections import Counter

from dt.dataset import partition


def entropy(labels: list[str]) -> float:
    """H(S) = − Σ p_k log₂ p_k.  Empty or pure → 0."""
    n = len(labels)
    if n == 0:
        return 0.0
    h = 0.0
    for count in Counter(labels).values():
        p = count / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def information_gain(
    rows: tuple[dict[str, str], ...],
    attr: str,
    target: str,
) -> float:
    """Gain(S, A) = H(S) − Σ (|Sv|/|S|) H(Sv)."""
    n = len(rows)
    if n == 0:
        return 0.0
    h = entropy([row[target] for row in rows])
    residual = 0.0
    for subset in partition(rows, attr).values():
        residual += len(subset) / n * entropy([row[target] for row in subset])
    return h - residual


def best_attribute(
    rows: tuple[dict[str, str], ...],
    attributes: tuple[str, ...],
    target: str,
) -> tuple[str, dict[str, float]]:
    """Return (best attr, all gains). Ties keep YAML order (first max)."""
    gains = {attr: information_gain(rows, attr, target) for attr in attributes}
    best = max(attributes, key=lambda a: (gains[a], -attributes.index(a)))
    return best, gains
