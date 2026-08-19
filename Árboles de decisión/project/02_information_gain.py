#!/usr/bin/env python3
"""Program 2: entropy of S and information gain of every attribute (root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dt.cli import build_parser, load  # noqa: E402
from dt.format import format_gains  # noqa: E402


def main() -> None:
    parser = build_parser("Compute H(S) and Gain(S, A) for each attribute at the root.")
    args = parser.parse_args()
    data = load(args)
    print(data.name)
    print()
    print("Gain(S, A) = H(S) − Σ (|Sv|/|S|) H(Sv)")
    print("ID3 picks the largest gain as the root test.")
    print()
    print(format_gains(data))


if __name__ == "__main__":
    main()
