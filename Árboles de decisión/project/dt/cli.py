"""CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from dt.dataset import Dataset, load_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "tennis.yaml"


def build_parser(description: str, with_query: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YAML example table")
    if with_query:
        parser.add_argument(
            "--query",
            nargs="*",
            default=None,
            help="Attribute=value pairs, e.g. Outlook=Sunny Humidity=High",
        )
    return parser


def load(args: argparse.Namespace) -> Dataset:
    return load_dataset(args.data)


def parse_query(items: list[str] | None) -> dict[str, str]:
    query: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"query must look like Outlook=Sunny, got {item!r}")
        key, value = item.split("=", 1)
        query[key.strip()] = value.strip()
    return query
