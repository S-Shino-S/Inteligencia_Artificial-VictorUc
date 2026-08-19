"""Pretty-print tables, gains, and ASCII trees."""

from __future__ import annotations

from dt.dataset import Dataset, format_counts, partition
from dt.entropy import entropy, information_gain
from dt.id3 import Leaf, Node, Split, classify


def format_dataset(data: Dataset) -> str:
    cols = list(data.attributes) + [data.target]
    widths = [max(len(c), *(len(row[c]) for row in data.examples)) for c in cols]
    num_w = max(2, len(str(len(data.examples))))
    header = f"{'#':>{num_w}}  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    lines = [data.name, "", header, "-" * len(header)]
    for i, row in enumerate(data.examples, 1):
        body = "  ".join(f"{row[c]:<{w}}" for c, w in zip(cols, widths))
        lines.append(f"{i:>{num_w}}  {body}")
    labels = data.labels()
    lines.append("")
    lines.append(f"target {data.target}:  {format_counts(labels)}   H(S) = {entropy(labels):.3f}")
    lines.append("attributes: " + ", ".join(data.attributes))
    return "\n".join(lines)


def format_gains(data: Dataset, rows: tuple[dict[str, str], ...] | None = None) -> str:
    rows = data.examples if rows is None else rows
    labels = [row[data.target] for row in rows]
    h = entropy(labels)
    ranked = sorted(
        ((information_gain(rows, attr, data.target), attr) for attr in data.attributes),
        key=lambda t: (-t[0], data.attributes.index(t[1])),
    )
    lines = [
        f"{len(rows)} examples   {format_counts(labels)}   H = {h:.3f}",
        "",
        f"{'attribute':<16}  {'gain':>8}  split",
        "-" * 72,
    ]
    best = ranked[0][1]
    for gain, attr in ranked:
        parts = []
        for value, subset in partition(rows, attr).items():
            lab = [row[data.target] for row in subset]
            parts.append(f"{value}:{len(subset)} [{format_counts(lab)}, H={entropy(lab):.3f}]")
        mark = "  ← best" if attr == best else ""
        lines.append(f"{attr:<16}  {gain:8.3f}  " + " · ".join(parts) + mark)
    return "\n".join(lines)


def _describe_split(node: Split) -> str:
    return f"{node.attribute}   (gain={node.gain:.3f}, n={node.n})"


def format_tree_root(node: Node) -> str:
    if isinstance(node, Leaf):
        return f"{node.label}   [{node.reason}, n={node.n}]"
    lines = [_describe_split(node)]
    items = list(node.branches.items())
    for i, (value, child) in enumerate(items):
        last = i == len(items) - 1
        lines.extend(_fmt_child(child, prefix="", is_last=last, edge=value))
    return "\n".join(lines)


def _fmt_child(node: Node, prefix: str, is_last: bool, edge: str) -> list[str]:
    branch = "└─ " if is_last else "├─ "
    ext = prefix + ("   " if is_last else "│  ")
    if isinstance(node, Leaf):
        return [f"{prefix}{branch}{edge} → {node.label}   [{node.reason}, n={node.n}]"]
    head = f"{prefix}{branch}{edge} → {_describe_split(node)}"
    lines = [head]
    items = list(node.branches.items())
    for i, (value, child) in enumerate(items):
        last = i == len(items) - 1
        lines.extend(_fmt_child(child, ext, last, value))
    return lines


def accuracy(node: Node, data: Dataset) -> float:
    ok = sum(1 for row in data.examples if classify(node, row)[0] == row[data.target])
    return ok / len(data.examples)
