#!/usr/bin/env python3
"""Program 2: print the full joint distribution (small boolean networks)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bn.cli import build_parser, load_bn  # noqa: E402
from bn.network import all_assignments, format_bool  # noqa: E402


def main() -> None:
    parser = build_parser("Enumerate P(world) for every assignment of the boolean variables.")
    args = parser.parse_args()
    bn = load_bn(args)
    n = len(bn.order)
    if n > 10:
        raise SystemExit(f"{n} variables → 2^{n} worlds; use a smaller network.")

    print(bn.name)
    print("Joint distribution  P(x1, …, xn) = Π P(xi | parents(xi))")
    print()
    header = "  ".join(f"{vid:>4}" for vid in bn.order) + "    P(world)"
    print(header)
    print("-" * len(header))
    total = 0.0
    for assignment in all_assignments(bn.order):
        p = bn.joint(assignment)
        total += p
        cells = "  ".join(f"{format_bool(assignment[vid]):>4}" for vid in bn.order)
        print(f"{cells}    {p:.6f}")
    print("-" * len(header))
    pad = " " * (len(header) - len("sum") - len(f"{total:.6f}"))
    print(f"sum{pad}{total:.6f}")


if __name__ == "__main__":
    main()
