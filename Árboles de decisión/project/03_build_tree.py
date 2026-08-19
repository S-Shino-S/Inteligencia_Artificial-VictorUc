#!/usr/bin/env python3
"""Program 3: grow an ID3 tree and print every split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dt.cli import build_parser, load  # noqa: E402
from dt.format import accuracy, format_tree_root  # noqa: E402
from dt.id3 import BuildEvent, build_tree  # noqa: E402


def main() -> None:
    parser = build_parser("Build a decision tree with ID3 (information gain).")
    args = parser.parse_args()
    data = load(args)
    events: list[BuildEvent] = []
    tree = build_tree(data, events=events)

    print(data.name)
    print()
    print("ID3 construction")
    print("-" * 72)
    for event in events:
        print(event.message)
    print()
    print("Tree")
    print("-" * 72)
    print(format_tree_root(tree))
    print()
    acc = accuracy(tree, data)
    print(f"Training accuracy: {acc:.0%}  ({len(data.examples)} examples)")


if __name__ == "__main__":
    main()
