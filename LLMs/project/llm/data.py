"""A tiny language problem loaded from YAML (corpus, n-gram, attention example)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm.tokenize import tokenize


@dataclass(frozen=True)
class AttentionEx:
    tokens: tuple[str, ...]
    alpha: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class LogitsDemo:
    labels: tuple[str, ...]
    z: tuple[float, ...]


@dataclass(frozen=True)
class Language:
    name: str
    vocab: tuple[str, ...]
    sentences: tuple[tuple[str, ...], ...]
    n: int
    probe: tuple[str, ...]
    stop: str
    seed: int
    lr: float
    epochs: int
    max_new: int
    temperature: float
    attention: AttentionEx | None
    logits_demo: LogitsDemo | None
    source: Path | None = None

    @property
    def context_len(self) -> int:
        return self.n - 1

    def token_id(self, tok: str) -> int:
        try:
            return self.vocab.index(tok)
        except ValueError as exc:
            raise ValueError(f"token {tok!r} is not in vocab {list(self.vocab)}") from exc

    def ids(self, tokens: tuple[str, ...] | list[str]) -> list[int]:
        return [self.token_id(t) for t in tokens]


def load_language(path: str | Path) -> Language:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_language(raw, source=path)


def parse_language(raw: dict[str, Any], source: Path | None = None) -> Language:
    vocab = tuple(str(t) for t in (raw.get("vocab") or []))
    if not vocab:
        raise ValueError("vocab: list the tokens in order")
    seen = set()
    for t in vocab:
        if t in seen:
            raise ValueError(f"duplicate vocab token {t!r}")
        seen.add(t)

    sentences = _sentences(raw, vocab)
    n = int(raw.get("n", 3))
    if n < 2:
        raise ValueError("n must be >= 2 (bigram = 2, trigram = 3)")

    probe_raw = raw.get("probe")
    if probe_raw is None:
        probe = sentences[0][: n - 1] if sentences else ()
    elif isinstance(probe_raw, str):
        probe = tuple(tokenize(probe_raw))
    else:
        probe = tuple(str(t) for t in probe_raw)
    for t in probe:
        if t not in seen:
            raise ValueError(f"probe token {t!r} is not in vocab")

    attn = _attention(raw.get("attention"))
    demo = _logits_demo(raw.get("logits_demo"))

    return Language(
        name=str(raw.get("name") or "language"),
        vocab=vocab,
        sentences=sentences,
        n=n,
        probe=probe,
        stop=str(raw.get("stop", ".")),
        seed=int(raw.get("seed", 0)),
        lr=float(raw.get("lr", 0.5)),
        epochs=int(raw.get("epochs", 200)),
        max_new=int(raw.get("max_new", 8)),
        temperature=float(raw.get("temperature", 1.0)),
        attention=attn,
        logits_demo=demo,
        source=source,
    )


def _sentences(raw: dict[str, Any], vocab: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    rows = raw.get("corpus") or []
    if not rows:
        raise ValueError("corpus: list of sentences (strings or token lists)")
    allowed = set(vocab)
    out: list[tuple[str, ...]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            toks = tuple(tokenize(row))
        else:
            toks = tuple(str(t) for t in row)
        if not toks:
            raise ValueError(f"corpus item {i} is empty")
        for t in toks:
            if t not in allowed:
                raise ValueError(f"corpus item {i}: {t!r} is not in vocab")
        out.append(toks)
    return tuple(out)


def _attention(raw: Any) -> AttentionEx | None:
    if not raw:
        return None
    tokens = tuple(str(t) for t in raw["tokens"])
    alpha = tuple(float(x) for x in raw["alpha"])
    values = tuple(tuple(float(x) for x in row) for row in raw["values"])
    if not (len(tokens) == len(alpha) == len(values)):
        raise ValueError("attention: tokens, alpha, values must have the same length")
    dim = len(values[0])
    for row in values:
        if len(row) != dim:
            raise ValueError("attention: value vectors must have the same length")
    scores = None
    if raw.get("scores") is not None:
        scores = tuple(float(x) for x in raw["scores"])
        if len(scores) != len(alpha):
            raise ValueError("attention: scores must match alpha")
    return AttentionEx(tokens=tokens, alpha=alpha, values=values, scores=scores)


def _logits_demo(raw: Any) -> LogitsDemo | None:
    if not raw:
        return None
    labels = tuple(str(t) for t in raw["labels"])
    z = tuple(float(x) for x in raw["z"])
    if len(labels) != len(z):
        raise ValueError("logits_demo: labels and z must have the same length")
    return LogitsDemo(labels=labels, z=z)
