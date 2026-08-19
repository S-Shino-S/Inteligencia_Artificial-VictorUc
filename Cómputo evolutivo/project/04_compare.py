#!/usr/bin/env python3
"""Program 4: same evaluation budget, GA vs uniform random search."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga.algorithm import random_search, run_ga  # noqa: E402
from ga.cli import ROOT as PROJECT_ROOT  # noqa: E402
from ga.cli import build_parser, load, make_rng  # noqa: E402
from ga.problem import format_problem  # noqa: E402
from ga.report import format_x  # noqa: E402

DEFAULT_COMPARE = PROJECT_ROOT / "problems" / "himmelblau.yaml"


def main() -> None:
    parser = build_parser(
        "Compare the GA with random search (same number of f evaluations).",
        with_run=True,
    )
    parser.set_defaults(problem=DEFAULT_COMPARE)
    args = parser.parse_args()
    problem = load(args)
    rng = make_rng(problem, args.seed)
    gens = args.generations if args.generations is not None else problem.generations

    print(format_problem(problem))
    print()

    ga = run_ga(problem, rng=rng, generations=gens, use_replay=False)
    rnd_rng = random.Random(rng.randint(0, 2**30))
    rnd = random_search(problem, ga.evaluations, rnd_rng)

    word = "higher" if problem.sense == "maximize" else "lower"
    print(f"Budget: {ga.evaluations} evaluations each  (want {word} f)")
    print("-" * 72)
    print(f"GA best      f = {ga.best_ever.objective:.6g}   {format_x(problem, ga.best_ever)}")
    print(f"Random best  f = {rnd.objective:.6g}   {problem.encoding.format_x(rnd.x)}")
    if problem.optimum_f is not None:
        print(f"Known opt    f = {problem.optimum_f:g}")
    print()
    print("The GA reuses bits from good parents. Random search does not.")


if __name__ == "__main__":
    main()
