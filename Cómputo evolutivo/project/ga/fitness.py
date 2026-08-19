"""Turn the objective f(x) into a non-negative fitness for selection.

Roulette needs larger fitness = more copies. Minimization therefore
inverts the current population's f values.
"""

from __future__ import annotations


def selection_fitness(objectives: list[float], sense: str) -> list[float]:
    if sense not in {"maximize", "minimize"}:
        raise ValueError("sense must be maximize or minimize")
    if not objectives:
        return []
    lo, hi = min(objectives), max(objectives)
    eps = 1e-12
    if sense == "maximize":
        # Use f itself when it is already a usable weight (lecture roulette).
        if lo < 0:
            return [f - lo + eps for f in objectives]
        return [max(f, 0.0) + eps for f in objectives]
    # minimize: low f → high fitness
    return [hi - f + eps for f in objectives]


def is_better(a: float, b: float, sense: str) -> bool:
    return a > b if sense == "maximize" else a < b
