"""CLI helpers."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm.data import Language, load_language

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATO = ROOT / "data" / "gato.yaml"
DEFAULT_MORE = ROOT / "data" / "more.yaml"


def build_parser(description: str, default: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=default or DEFAULT_GATO, help="YAML language")
    parser.add_argument("--seed", type=int, default=None, help="Override YAML seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override YAML epochs")
    parser.add_argument("--temperature", type=float, default=None, help="Override YAML T")
    return parser


def load(args: argparse.Namespace) -> Language:
    return load_language(args.data)


def make_rng(lang: Language, seed_override: int | None) -> random.Random:
    seed = seed_override if seed_override is not None else lang.seed
    return random.Random(seed)
