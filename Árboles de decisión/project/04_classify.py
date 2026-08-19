#!/usr/bin/env python3
"""Program 4: classify one example and print the path down the tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dt.cli import build_parser, load, parse_query  # noqa: E402
from dt.format import format_tree_root  # noqa: E402
from dt.id3 import build_tree, classify  # noqa: E402


def main() -> None:
    parser = build_parser("Classify an example by walking the ID3 tree.", with_query=True)
    args = parser.parse_args()
    data = load(args)
    if args.query is not None:
        query = parse_query(args.query)
    else:
        query = dict(data.query)
    if not query:
        raise SystemExit("No query: set query: in the YAML file or pass --query Attr=value ...")

    tree = build_tree(data)
    label, path = classify(tree, query)

    print(data.name)
    print()
    print("Tree")
    print("-" * 72)
    print(format_tree_root(tree))
    print()
    print("Query")
    print("-" * 72)
    print("  " + ", ".join(f"{k}={v}" for k, v in query.items()))
    print()
    print("Path")
    print("-" * 72)
    for step in path:
        print(f"  {step}")
    print()
    print(f"Predicted {data.target}: {label}")


if __name__ == "__main__":
    main()
