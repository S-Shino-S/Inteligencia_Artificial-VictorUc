"""ID3: greedy tree by information gain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from dt.dataset import Dataset, majority, partition
from dt.entropy import best_attribute, entropy


@dataclass
class Leaf:
    label: str
    n: int
    reason: str


@dataclass
class Split:
    attribute: str
    gain: float
    n: int
    default: str
    branches: dict[str, Node] = field(default_factory=dict)


Node = Union[Leaf, Split]


@dataclass
class BuildEvent:
    message: str


def is_pure(labels: list[str]) -> bool:
    return len(set(labels)) <= 1


def id3(
    rows: tuple[dict[str, str], ...],
    attributes: tuple[str, ...],
    target: str,
    default: str,
    events: list[BuildEvent] | None = None,
    depth: int = 0,
) -> Node:
    indent = "  " * depth
    log = events.append if events is not None else lambda _e: None
    labels = [row[target] for row in rows]
    n = len(rows)

    if n == 0:
        log(BuildEvent(f"{indent}no examples → leaf {default} (parent majority)"))
        return Leaf(default, 0, "no examples")
    if is_pure(labels):
        log(BuildEvent(f"{indent}pure {n}× {labels[0]} → leaf {labels[0]}"))
        return Leaf(labels[0], n, "pure")
    if not attributes:
        maj = majority(labels)
        log(BuildEvent(f"{indent}no attributes left → leaf {maj} (majority of {n})"))
        return Leaf(maj, n, "majority")

    h = entropy(labels)
    attr, gains = best_attribute(rows, attributes, target)
    gain_s = "  ".join(f"{a}={gains[a]:.3f}" for a in attributes)
    log(BuildEvent(f"{indent}n={n}  H={h:.3f}  {gain_s}  → split {attr} ({gains[attr]:.3f})"))
    rest = tuple(a for a in attributes if a != attr)
    parent_maj = majority(labels)
    node = Split(attribute=attr, gain=gains[attr], n=n, default=parent_maj)
    groups = partition(rows, attr)
    # Stable branch order: first-seen in the data, then any leftover values.
    for value, subset in groups.items():
        log(BuildEvent(f"{indent}{attr} = {value}  ({len(subset)} examples)"))
        node.branches[value] = id3(subset, rest, target, parent_maj, events, depth + 1)
    return node


def build_tree(data: Dataset, events: list[BuildEvent] | None = None) -> Node:
    labels = data.labels()
    default = majority(labels)
    if events is not None:
        events.append(BuildEvent(f"target {data.target}   {len(data.examples)} examples"))
    return id3(data.examples, data.attributes, data.target, default, events)


def classify(node: Node, example: dict[str, str]) -> tuple[str, list[str]]:
    """Return (predicted label, path of tests)."""
    path: list[str] = []
    current = node
    while isinstance(current, Split):
        value = example.get(current.attribute)
        if value is None:
            path.append(f"{current.attribute} missing → majority {current.default}")
            return current.default, path
        path.append(f"{current.attribute} = {value}")
        child = current.branches.get(value)
        if child is None:
            path.append(f"(unseen value, majority {current.default})")
            return current.default, path
        current = child
    path.append(f"leaf {current.label}  ({current.reason}, n={current.n})")
    return current.label, path
