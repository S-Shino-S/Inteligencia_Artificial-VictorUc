#!/usr/bin/env python3
"""Program 3: P(query | evidence) by enumeration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bn.cli import build_parser, load_query_problem  # noqa: E402
from bn.enumerate import enumerate_ask, format_distribution  # noqa: E402
from bn.network import format_assignment  # noqa: E402


def main() -> None:
    parser = build_parser(
        "Exact inference by enumeration: P(query | evidence).",
        with_query=True,
    )
    args = parser.parse_args()
    bn, query, evidence = load_query_problem(args)
    dist = enumerate_ask(bn, query, evidence)
    print(bn.name)
    print(f"Query {query}   evidence: {format_assignment(evidence) if evidence else '(none)'}")
    print()
    print(format_distribution(query, dist, evidence))


if __name__ == "__main__":
    main()
