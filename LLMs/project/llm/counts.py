"""Count-based n-grams: P(w_t | w_{t-n+1} … w_{t-1})."""

from __future__ import annotations

from collections import defaultdict

from llm.data import Language


def context_key(tokens: tuple[str, ...] | list[str], n: int) -> tuple[str, ...]:
    width = n - 1
    if width <= 0:
        return ()
    if len(tokens) >= width:
        return tuple(tokens[-width:])
    return tuple(tokens)


def count_ngrams(lang: Language) -> dict[tuple[str, ...], list[int]]:
    table: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0] * len(lang.vocab))
    width = lang.context_len
    for sent in lang.sentences:
        if len(sent) <= width:
            continue
        for t in range(width, len(sent)):
            ctx = tuple(sent[t - width : t])
            table[ctx][lang.token_id(sent[t])] += 1
    return dict(table)


def dist_from_counts(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        n = len(counts)
        return [1.0 / n] * n
    return [c / total for c in counts]


def next_dist(
    lang: Language,
    table: dict[tuple[str, ...], list[int]],
    context: tuple[str, ...] | list[str],
) -> list[float]:
    key = context_key(context, lang.n)
    if key not in table:
        return [1.0 / len(lang.vocab)] * len(lang.vocab)
    return dist_from_counts(table[key])
