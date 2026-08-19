#!/usr/bin/env python3
"""Program 1: print the GA problem (function, encoding, parameters)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga.cli import build_parser, load  # noqa: E402
from ga.functions import FUNCTIONS  # noqa: E402
from ga.problem import format_problem  # noqa: E402


def main() -> None:
    parser = build_parser("Show a YAML genetic-algorithm problem.")
    args = parser.parse_args()
    problem = load(args)
    print(format_problem(problem))
    print()
    print("Built-in functions: " + ", ".join(sorted(FUNCTIONS)))
    print("Edit the YAML file, then rerun this program.")


if __name__ == "__main__":
    main()
