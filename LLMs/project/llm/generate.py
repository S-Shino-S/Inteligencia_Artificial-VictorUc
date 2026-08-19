"""Autoregressive loop: append one token, repeat."""

from __future__ import annotations

import random
from collections.abc import Callable

from llm.counts import context_key
from llm.data import Language
from llm.math import argmax, log_probs, sample_index, softmax

DistFn = Callable[[tuple[str, ...]], list[float]]


def generate(
    lang: Language,
    dist_fn: DistFn,
    prefix: tuple[str, ...] | list[str] | None = None,
    max_new: int | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random.Random(lang.seed)
    tokens = list(prefix if prefix is not None else lang.probe)
    steps = lang.max_new if max_new is None else max_new
    for _ in range(steps):
        ctx = context_key(tokens, lang.n)
        probs = dist_fn(ctx)
        if greedy:
            idx = argmax(probs)
        else:
            if temperature != 1.0:
                probs = softmax(log_probs(probs), temperature=temperature)
            idx = sample_index(probs, rng)
        tok = lang.vocab[idx]
        tokens.append(tok)
        if tok == lang.stop:
            break
    return tokens
