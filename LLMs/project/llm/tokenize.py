"""Whitespace tokenizer: letters stay together, punctuation is its own token."""

from __future__ import annotations

PUNCT = set(".,!?;:")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in text.strip().lower():
        if ch.isspace():
            flush()
        elif ch in PUNCT:
            flush()
            tokens.append(ch)
        else:
            buf.append(ch)
    flush()
    return tokens
