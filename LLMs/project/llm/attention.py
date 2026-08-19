"""Causal self-attention (dot-product scores) and the lecture weighted sum."""

from __future__ import annotations

from llm.math import dot, mix, softmax


def attend(
    alpha: list[float] | tuple[float, ...],
    values: list[list[float]] | tuple[tuple[float, ...], ...],
) -> list[float]:
    return mix(list(alpha), values)


def causal_self_attention(
    vectors: list[list[float]] | tuple[tuple[float, ...], ...],
    temperature: float = 1.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Each position t only looks at 0..t. Score = x_t · x_j."""
    n = len(vectors)
    alphas: list[list[float]] = []
    outs: list[list[float]] = []
    for t in range(n):
        scores = [dot(vectors[t], vectors[j]) for j in range(t + 1)]
        a = softmax(scores, temperature=temperature)
        a_full = a + [0.0] * (n - t - 1)
        alphas.append(a_full)
        outs.append(mix(a, [list(map(float, vectors[j])) for j in range(t + 1)]))
    return outs, alphas
