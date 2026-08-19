"""Boolean Bayesian network: DAG + CPTs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Variable:
    vid: str
    label: str

    def __str__(self) -> str:
        return f"{self.vid} ({self.label})" if self.label != self.vid else self.vid


class BayesianNetwork:
    """Boolean BN. Each CPT stores P(X = True | parent assignment)."""

    def __init__(
        self,
        variables: list[Variable],
        parents: dict[str, list[str]],
        cpts: dict[str, dict[tuple[bool, ...], float]],
        name: str = "",
    ) -> None:
        self.name = name
        self.variables = {v.vid: v for v in variables}
        self.order = [v.vid for v in variables]
        self.parents = {vid: list(parents.get(vid, [])) for vid in self.order}
        self.cpts = cpts
        self._check()

    def _check(self) -> None:
        ids = set(self.order)
        if len(ids) != len(self.order):
            raise ValueError("duplicate variable id")
        for vid, pars in self.parents.items():
            for p in pars:
                if p not in ids:
                    raise ValueError(f"unknown parent {p} of {vid}")
        order = topological_order(self.parents)
        if set(order) != ids:
            raise ValueError("the graph has a cycle or a dangling node")
        self.order = order
        for vid in self.order:
            pars = self.parents[vid]
            expected = list(product([False, True], repeat=len(pars))) if pars else [()]
            table = self.cpts.get(vid)
            if table is None:
                raise ValueError(f"missing CPT for {vid}")
            for key in expected:
                if key not in table:
                    raise ValueError(f"CPT {vid} missing parent assignment {key}")
                p = table[key]
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"P({vid}=true | {key}) = {p} is not a probability")

    def p_true(self, vid: str, assignment: dict[str, bool]) -> float:
        key = tuple(assignment[p] for p in self.parents[vid])
        return self.cpts[vid][key]

    def p_value(self, vid: str, value: bool, assignment: dict[str, bool]) -> float:
        pt = self.p_true(vid, assignment)
        return pt if value else 1.0 - pt

    def joint(self, assignment: dict[str, bool]) -> float:
        """P(full assignment) = Π P(xi | parents(xi))."""
        return prod(self.p_value(vid, assignment[vid], assignment) for vid in self.order)


def prod(values) -> float:
    out = 1.0
    for v in values:
        out *= v
    return out


def topological_order(parents: dict[str, list[str]]) -> list[str]:
    remaining = {n: set(pars) for n, pars in parents.items()}
    # YAML / insertion order, so independent roots print as in the file.
    ready = [n for n, pars in remaining.items() if not pars]
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child, pars in remaining.items():
            if node in pars:
                pars.remove(node)
                if not pars and child not in order and child not in ready:
                    ready.append(child)
    return order


def all_assignments(var_ids: list[str]) -> list[dict[str, bool]]:
    rows = []
    for bits in product([False, True], repeat=len(var_ids)):
        rows.append(dict(zip(var_ids, bits)))
    return rows


def format_bool(value: bool) -> str:
    return "t" if value else "f"


def format_assignment(assignment: dict[str, bool]) -> str:
    return ", ".join(f"{k}={format_bool(v)}" for k, v in assignment.items())
