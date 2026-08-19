"""One softmax per n-gram context (the last layer of a tiny LM)."""

from __future__ import annotations

from collections import defaultdict

from llm.counts import context_key
from llm.data import Language
from llm.math import nll, softmax


def examples(lang: Language) -> list[tuple[tuple[str, ...], int]]:
    width = lang.context_len
    out: list[tuple[tuple[str, ...], int]] = []
    for sent in lang.sentences:
        if len(sent) <= width:
            continue
        for t in range(width, len(sent)):
            ctx = tuple(sent[t - width : t])
            out.append((ctx, lang.token_id(sent[t])))
    return out


def train_logits(lang: Language) -> tuple[dict[tuple[str, ...], list[float]], list[float]]:
    """Batch SGD on −log P. Gradient of softmax+NLL is P − one-hot(y)."""
    table: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0] * len(lang.vocab))
    pairs = examples(lang)
    if not pairs:
        raise ValueError("corpus is shorter than n; need at least one next-token pair")
    history: list[float] = []
    v = len(lang.vocab)
    for _ in range(lang.epochs):
        grads: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0] * v)
        n_ctx: dict[tuple[str, ...], int] = defaultdict(int)
        total = 0.0
        for ctx, y in pairs:
            z = table[ctx]
            p = softmax(z)
            total += nll(p, y)
            n_ctx[ctx] += 1
            for i in range(v):
                grads[ctx][i] += p[i] - (1.0 if i == y else 0.0)
        for ctx, g in grads.items():
            z = table[ctx]
            n = n_ctx[ctx]
            for i in range(v):
                z[i] -= lang.lr * g[i] / n
        history.append(total / len(pairs))
    return dict(table), history


def next_dist_logits(
    lang: Language,
    table: dict[tuple[str, ...], list[float]],
    context: tuple[str, ...] | list[str],
    temperature: float = 1.0,
) -> list[float]:
    key = context_key(context, lang.n)
    z = table.get(key)
    if z is None:
        return [1.0 / len(lang.vocab)] * len(lang.vocab)
    return softmax(z, temperature=temperature)
