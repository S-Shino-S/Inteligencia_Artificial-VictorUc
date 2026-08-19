"""Labeled examples loaded from YAML."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Dataset:
    name: str
    target: str
    attributes: tuple[str, ...]
    examples: tuple[dict[str, str], ...]
    query: dict[str, str]
    source: Path | None = None

    def labels(self, rows: tuple[dict[str, str], ...] | None = None) -> list[str]:
        rows = self.examples if rows is None else rows
        return [row[self.target] for row in rows]

    def values(self, attr: str, rows: tuple[dict[str, str], ...] | None = None) -> list[str]:
        rows = self.examples if rows is None else rows
        return [row[attr] for row in rows]


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_dataset(raw, source=path)


def parse_dataset(raw: dict[str, Any], source: Path | None = None) -> Dataset:
    target = str(raw.get("target") or "Play")
    attributes = tuple(str(a) for a in (raw.get("attributes") or []))
    if not attributes:
        raise ValueError("attributes: list the input columns (not the target)")
    if target in attributes:
        raise ValueError("target must not also appear in attributes")
    examples = []
    for i, row in enumerate(raw.get("examples") or [], 1):
        item = {str(k): _as_str(v) for k, v in row.items()}
        for attr in attributes:
            if attr not in item:
                raise ValueError(f"example {i} is missing {attr}")
        if target not in item:
            raise ValueError(f"example {i} is missing target {target}")
        examples.append(item)
    if not examples:
        raise ValueError("examples: need at least one row")
    query = {str(k): _as_str(v) for k, v in (raw.get("query") or {}).items()}
    return Dataset(
        name=str(raw.get("name") or "Decision tree data"),
        target=target,
        attributes=attributes,
        examples=tuple(examples),
        query=query,
        source=source,
    )


def _as_str(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "None"
    return str(value)


def partition(rows: tuple[dict[str, str], ...], attr: str) -> dict[str, tuple[dict[str, str], ...]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[attr], []).append(row)
    return {k: tuple(v) for k, v in groups.items()}


def majority(labels: list[str]) -> str:
    if not labels:
        raise ValueError("majority of an empty label list")
    counts = Counter(labels)
    # Tie: alphabetical so runs are deterministic.
    best = max(counts.values())
    return sorted(c for c, n in counts.items() if n == best)[0]


def format_counts(labels: list[str]) -> str:
    c = Counter(labels)
    if not c:
        return "(none)"
    return ", ".join(f"{n} {lab}" for lab, n in sorted(c.items()))
