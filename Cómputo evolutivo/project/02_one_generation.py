#!/usr/bin/env python3
"""Program 2: one generation, with selection / crossover / mutation printed.

Default problem is Goldberg's x² lecture example (fixed population + replay).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga.algorithm import run_ga  # noqa: E402
from ga.cli import build_parser, load, make_rng  # noqa: E402
from ga.problem import format_problem  # noqa: E402
from ga.report import format_population  # noqa: E402


def main() -> None:
    parser = build_parser("Trace a single GA generation (operators visible).")
    args = parser.parse_args()
    problem = load(args)
    rng = make_rng(problem, args.seed)

    print(format_problem(problem))
    print()

    result = run_ga(
        problem,
        rng=rng,
        generations=1,
        verbose=True,
        use_replay=problem.replay is not None,
    )
    gen0 = result.history[0].population
    gen1 = result.history[1].population

    print(format_population(problem, gen0, "Generation 0"))
    print()
    print("Operators")
    print("-" * 72)
    if result.events:
        for event in result.events:
            print(event.message)
    else:
        print("(no operator log)")
    print()
    print(format_population(problem, gen1, "Generation 1"))
    print()
    f0 = result.history[0]
    f1 = result.history[1]
    print("Summary")
    print("-" * 72)
    print(f"avg f:   {f0.average:.4f}  →  {f1.average:.4f}")
    print(f"best f:  {f0.best.objective:.4f}  →  {f1.best.objective:.4f}")
    print(f"worst f: {f0.worst.objective:.4f}  →  {f1.worst.objective:.4f}")


if __name__ == "__main__":
    main()
