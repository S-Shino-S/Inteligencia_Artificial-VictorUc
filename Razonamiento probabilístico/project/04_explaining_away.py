#!/usr/bin/env python3
"""Program 4: explaining away on the wet-grass network.

Wet grass (P=t) raises P(rain). Observing the sprinkler (A=t) as well
explains the wet grass and lowers P(rain) again.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bn.cli import build_parser, load_bn  # noqa: E402
from bn.enumerate import enumerate_ask, format_distribution  # noqa: E402


def main() -> None:
    parser = build_parser("Show explaining away: P(L | P=t) vs P(L | P=t, A=t).")
    args = parser.parse_args()
    bn = load_bn(args)
    needed = {"L", "A", "P"}
    if not needed.issubset(bn.variables):
        raise SystemExit(
            "This program needs variables L, A, and P (wet-grass network). "
            "Use the default --network networks/wet_grass.yaml"
        )

    prior = enumerate_ask(bn, "L", {})
    given_wet = enumerate_ask(bn, "L", {"P": True})
    given_wet_and_sprinkler = enumerate_ask(bn, "L", {"P": True, "A": True})

    print(bn.name)
    print()
    print("Prior")
    print(format_distribution("L", prior))
    print()
    print("Wet grass only — rain becomes more likely")
    print(format_distribution("L", given_wet, {"P": True}))
    print()
    print("Wet grass and sprinkler on — the sprinkler explains the grass")
    print(format_distribution("L", given_wet_and_sprinkler, {"P": True, "A": True}))
    print()
    print(
        f"P(L=t) = {prior[True]:.3f}  →  "
        f"P(L=t | P=t) = {given_wet[True]:.3f}  →  "
        f"P(L=t | P=t, A=t) = {given_wet_and_sprinkler[True]:.3f}"
    )


if __name__ == "__main__":
    main()
