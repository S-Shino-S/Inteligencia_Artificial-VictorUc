"""CLI helpers."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ga.problem import Problem, load_problem

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBLEM = ROOT / "problems" / "goldberg_x2.yaml"


def build_parser(description: str, with_run: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM, help="YAML GA problem")
    parser.add_argument("--seed", type=int, default=None, help="Override the YAML seed")
    if with_run:
        parser.add_argument("--generations", type=int, default=None, help="Override YAML generations")
    return parser


def load(args: argparse.Namespace) -> Problem:
    return load_problem(args.problem)


def make_rng(problem: Problem, seed_override: int | None) -> random.Random:
    seed = seed_override if seed_override is not None else problem.seed
    return random.Random(seed)
