"""CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from bn.load import load_default_query, load_network
from bn.network import BayesianNetwork

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NET = ROOT / "networks" / "wet_grass.yaml"


def build_parser(description: str, with_query: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--network", type=Path, default=DEFAULT_NET, help="YAML Bayesian network")
    if with_query:
        parser.add_argument("--query", default=None, help="Query variable, e.g. L")
        parser.add_argument(
            "--evidence",
            nargs="*",
            default=None,
            help="Evidence as Var=true/false, e.g. P=true A=false",
        )
    return parser


def parse_evidence(items: list[str] | None) -> dict[str, bool]:
    evidence: dict[str, bool] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"evidence must look like L=true, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        text = raw.strip().lower()
        if text in {"t", "true", "1", "yes"}:
            evidence[key] = True
        elif text in {"f", "false", "0", "no"}:
            evidence[key] = False
        else:
            raise SystemExit(f"value must be true/false, got {raw!r}")
    return evidence


def load_bn(args: argparse.Namespace) -> BayesianNetwork:
    return load_network(args.network)


def load_query_problem(args: argparse.Namespace) -> tuple[BayesianNetwork, str, dict[str, bool]]:
    bn = load_bn(args)
    yaml_query, yaml_evidence = load_default_query(args.network)
    query = args.query or yaml_query
    if args.evidence is not None:
        evidence = parse_evidence(args.evidence)
    else:
        evidence = yaml_evidence
    if not query:
        raise SystemExit("No query: set query.variable in the YAML file or pass --query")
    return bn, query, evidence
