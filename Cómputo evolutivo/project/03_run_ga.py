#!/usr/bin/env python3
"""Program 3: run the simple GA for several generations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga.algorithm import run_ga  # noqa: E402
from ga.cli import build_parser, load, make_rng  # noqa: E402
from ga.problem import format_problem  # noqa: E402
from ga.report import format_best, format_history, sparkline  # noqa: E402


def main() -> None:
    parser = build_parser("Run the simple genetic algorithm.", with_run=True)
    args = parser.parse_args()
    problem = load(args)
    rng = make_rng(problem, args.seed)
    gens = args.generations if args.generations is not None else problem.generations

    print(format_problem(problem))
    if args.generations is not None:
        print(f"Override:    generations = {gens}")
    print()

    result = run_ga(problem, rng=rng, generations=gens, verbose=False, use_replay=False)
    print(format_history(problem, result))
    print()
    values = [rec.best.objective for rec in result.history]
    label = "best f over generations"
    print(f"{label}:  {sparkline(values)}")
    print()
    print(format_best(problem, result))


if __name__ == "__main__":
    main()
