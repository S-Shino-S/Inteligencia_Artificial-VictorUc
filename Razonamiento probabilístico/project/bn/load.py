"""Load a boolean Bayesian network from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bn.network import BayesianNetwork, Variable, format_bool


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_network(path: str | Path) -> BayesianNetwork:
    return parse_network(load_yaml(path))


def load_default_query(path: str | Path) -> tuple[str | None, dict[str, bool]]:
    """Optional YAML block: query.variable and query.evidence."""
    raw = load_yaml(path).get("query") or {}
    variable = raw.get("variable")
    evidence: dict[str, bool] = {}
    for key, value in (raw.get("evidence") or {}).items():
        evidence[str(key)] = _as_bool(value)
    return (str(variable) if variable is not None else None, evidence)


def parse_network(raw: dict[str, Any]) -> BayesianNetwork:
    name = str(raw.get("name") or "Bayesian network")
    variables = []
    for item in raw.get("variables") or []:
        if isinstance(item, str):
            variables.append(Variable(item, item))
        else:
            variables.append(Variable(str(item["id"]), str(item.get("label") or item["id"])))
    ids = [v.vid for v in variables]
    parents_raw = raw.get("structure") or raw.get("parents") or {}
    parents = {vid: list(parents_raw.get(vid, [])) for vid in ids}

    cpts: dict[str, dict[tuple[bool, ...], float]] = {}
    cpt_raw = raw.get("cpts") or {}
    for vid in ids:
        spec = cpt_raw.get(vid)
        if spec is None:
            raise ValueError(f"cpts.{vid} is missing")
        if isinstance(spec, (int, float)):
            cpts[vid] = {(): float(spec)}
            continue
        table: dict[tuple[bool, ...], float] = {}
        pars = parents[vid]
        for row in spec:
            when = row.get("when") or {}
            key = tuple(_as_bool(when[p]) for p in pars)
            table[key] = float(row["p_true"])
        cpts[vid] = table
    return BayesianNetwork(variables, parents, cpts, name=name)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"t", "true", "1", "yes"}:
        return True
    if text in {"f", "false", "0", "no"}:
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def format_network(bn: BayesianNetwork) -> str:
    lines = [bn.name, ""]
    lines.append("Variables:")
    for vid in bn.order:
        pars = bn.parents[vid]
        par_s = ", ".join(pars) if pars else "(none)"
        lines.append(f"  {bn.variables[vid]}   parents: {par_s}")
    lines.append("")
    lines.append("CPTs  (values are P(X = t | parents)):")
    for vid in bn.order:
        pars = bn.parents[vid]
        table = bn.cpts[vid]
        if not pars:
            lines.append(f"  P({vid}=t) = {table[()]:.4f}")
            continue
        lines.append(f"  {vid} | " + " ".join(pars) + " | P(t)")
        for key, p in table.items():
            cond = " ".join(format_bool(b) for b in key)
            lines.append(f"       {cond} | {p:.4f}")
    return "\n".join(lines)
