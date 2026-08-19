"""Softmax, NLL, sampling — every exp is visible."""

from __future__ import annotations

import math
import random


def softmax(z: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [v / temperature for v in z]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def nll(probs: list[float], target: int) -> float:
    p = probs[target]
    if p <= 0:
        return 1e6
    return -math.log(p)


def log_probs(probs: list[float], eps: float = 1e-12) -> list[float]:
    return [math.log(p + eps) for p in probs]


def mix(alpha: list[float], values: list[list[float]] | tuple[tuple[float, ...], ...]) -> list[float]:
    if abs(sum(alpha) - 1.0) > 1e-6:
        raise ValueError(f"alpha must sum to 1 (got {sum(alpha):.6f})")
    if len(alpha) != len(values):
        raise ValueError("alpha and values must have the same length")
    dim = len(values[0])
    out = [0.0] * dim
    for a, vec in zip(alpha, values):
        for i in range(dim):
            out[i] += a * float(vec[i])
    return out


def dot(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def argmax(values: list[float]) -> int:
    """First index wins ties (YAML / vocab order)."""
    best_i, best_v = 0, values[0]
    for i, v in enumerate(values[1:], 1):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def sample_index(probs: list[float], rng: random.Random) -> int:
    u = rng.random()
    acc = 0.0
    last = len(probs) - 1
    for i, p in enumerate(probs):
        acc += p
        if u <= acc or i == last:
            return i
    return last
