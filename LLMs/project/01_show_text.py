#!/usr/bin/env python3
"""Program 1: tokenize the corpus and show n-gram counts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm.cli import build_parser, load  # noqa: E402
from llm.format import format_count_table, format_language  # noqa: E402


def main() -> None:
    parser = build_parser("Show tokens, ids, and P(next | context) from counts.")
    args = parser.parse_args()
    lang = load(args)
    print(format_language(lang))
    print()
    print(format_count_table(lang))
    print()
    print("The model does not see a sentence. It sees a list of ids from a fixed vocab.")
    print("Edit the YAML file, then rerun this program.")


if __name__ == "__main__":
    main()
