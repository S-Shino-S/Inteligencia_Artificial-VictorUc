"""Exact inference by enumeration (AIMA Figure 13.11 / 14.9).

Variables are visited in topological order. For each variable Y:
- if Y is already assigned (query or evidence), multiply by P(y | parents);
- otherwise sum over y ∈ {false, true}.
"""

from __future__ import annotations

from bn.network import BayesianNetwork, format_assignment, format_bool


def enumerate_ask(
    bn: BayesianNetwork,
    query: str,
    evidence: dict[str, bool],
) -> dict[bool, float]:
    """Return P(query | evidence) as {False: p, True: p}."""
    if query not in bn.variables:
        raise ValueError(f"unknown query variable {query}")
    for vid in evidence:
        if vid not in bn.variables:
            raise ValueError(f"unknown evidence variable {vid}")
    if query in evidence:
        raise ValueError("query variable cannot also appear in the evidence")
    dist: dict[bool, float] = {}
    for value in (False, True):
        assignment = dict(evidence)
        assignment[query] = value
        dist[value] = enumerate_all(bn, bn.order, assignment)
    return _normalize(dist)


def enumerate_all(
    bn: BayesianNetwork,
    var_ids: list[str],
    evidence: dict[str, bool],
) -> float:
    """Sum (or multiply) along the remaining variables, given evidence."""
    if not var_ids:
        return 1.0
    y_id, rest = var_ids[0], var_ids[1:]
    if y_id in evidence:
        return bn.p_value(y_id, evidence[y_id], evidence) * enumerate_all(bn, rest, evidence)
    total = 0.0
    for value in (False, True):
        extended = dict(evidence)
        extended[y_id] = value
        total += bn.p_value(y_id, value, extended) * enumerate_all(bn, rest, extended)
    return total


def _normalize(dist: dict[bool, float]) -> dict[bool, float]:
    z = sum(dist.values())
    if z == 0.0:
        raise ValueError("evidence has probability 0 under this network")
    return {k: v / z for k, v in dist.items()}


def format_distribution(
    query: str,
    dist: dict[bool, float],
    evidence: dict[str, bool] | None = None,
) -> str:
    ev = f" | {format_assignment(evidence)}" if evidence else ""
    lines = [f"P({query}{ev})"]
    for value in (False, True):
        lines.append(f"  {query}={format_bool(value)}   {dist[value]:.4f}")
    return "\n".join(lines)
