"""Simple-GA operators: roulette / tournament, one-point crossover, bit-flip."""

from __future__ import annotations

import random


def roulette(fitness: list[float], rng: random.Random) -> int:
    """Return an index; P(i) ∝ fitness[i]."""
    total = sum(fitness)
    if total <= 0:
        return rng.randrange(len(fitness))
    pick = rng.random() * total
    acc = 0.0
    for i, w in enumerate(fitness):
        acc += w
        if acc >= pick:
            return i
    return len(fitness) - 1


def tournament(fitness: list[float], rng: random.Random, k: int = 2) -> int:
    """Return the fittest of k random indices (fitness already oriented)."""
    k = max(1, min(k, len(fitness)))
    contestants = [rng.randrange(len(fitness)) for _ in range(k)]
    return max(contestants, key=lambda i: fitness[i])


def one_point_crossover(
    a: str, b: str, rng: random.Random, p_c: float
) -> tuple[str, str, int | None]:
    """Return children and the cut index, or (a, b, None) if no crossover."""
    if len(a) != len(b):
        raise ValueError("parents must have the same length")
    if len(a) < 2 or rng.random() >= p_c:
        return a, b, None
    cut = rng.randrange(1, len(a))
    return a[:cut] + b[cut:], b[:cut] + a[cut:], cut


def crossover_at(a: str, b: str, cut: int) -> tuple[str, str]:
    """Deterministic one-point crossover (lecture replay)."""
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def mutate(bits: str, rng: random.Random, p_m: float) -> tuple[str, list[int]]:
    """Flip each bit with probability p_m. Return new string and flipped indices."""
    chars = list(bits)
    flipped = []
    for i, ch in enumerate(chars):
        if rng.random() < p_m:
            chars[i] = "1" if ch == "0" else "0"
            flipped.append(i)
    return "".join(chars), flipped
