#!/usr/bin/env python3
"""Program 1: print the Bayesian network (DAG + CPTs)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bn.cli import build_parser, load_bn  # noqa: E402
from bn.load import format_network  # noqa: E402


def main() -> None:
    parser = build_parser("Show variables, parents, and CPTs from a YAML Bayesian network.")
    args = parser.parse_args()
    bn = load_bn(args)
    print(format_network(bn))
    print()
    print("Edit the YAML file, then rerun this program.")


if __name__ == "__main__":
    main()
