"""The corpus loaded from YAML: documents, example queries, and settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Document:
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Corpus:
    name: str
    documents: tuple[Document, ...]
    queries: tuple[str, ...]
    examples: tuple[str, ...]
    chunk_words: int
    overlap: int
    top_k: int
    min_score: float
    stopwords: frozenset[str]
    source: Path | None = None


def load_corpus(path: str | Path) -> Corpus:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_corpus(raw, source=path)


def parse_corpus(raw: dict[str, Any], source: Path | None = None) -> Corpus:
    documents = _documents(raw.get("documents"))
    if not documents:
        raise ValueError("documents: list at least one {title, source, text}")

    chunk_words = int(raw.get("chunk_words", 30))
    if chunk_words < 1:
        raise ValueError("chunk_words must be >= 1")
    overlap = int(raw.get("overlap", 6))
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")

    return Corpus(
        name=str(raw.get("name") or "corpus"),
        documents=documents,
        queries=tuple(str(q) for q in (raw.get("queries") or [])),
        examples=tuple(str(t) for t in (raw.get("examples") or [])),
        chunk_words=chunk_words,
        overlap=overlap,
        top_k=int(raw.get("top_k", 2)),
        min_score=float(raw.get("min_score", 0.05)),
        stopwords=frozenset(str(w).lower() for w in (raw.get("stopwords") or [])),
        source=source,
    )


def _documents(rows: Any) -> tuple[Document, ...]:
    out: list[Document] = []
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            raise ValueError(f"documents[{i}] must be a mapping with text")
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"documents[{i}] has no text")
        out.append(
            Document(
                title=str(row.get("title") or f"doc {i}"),
                source=str(row.get("source") or "unknown"),
                text=text,
            )
        )
    return tuple(out)
