#!/usr/bin/env python3
"""Program 1: print the example table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dt.cli import build_parser, load  # noqa: E402
from dt.format import format_dataset  # noqa: E402


def main() -> None:
    parser = build_parser("Show labeled examples from a YAML table.")
    args = parser.parse_args()
    data = load(args)
    print(format_dataset(data))
    print()
    print("Edit the YAML file, then rerun this program.")


if __name__ == "__main__":
    main()
