"""Pretty-print populations and run histories."""

from __future__ import annotations

from ga.algorithm import Individual, RunResult
from ga.problem import Problem


def format_x(problem: Problem, ind: Individual) -> str:
    return problem.encoding.format_x(ind.x)


def format_population(problem: Problem, pop: list[Individual], title: str = "") -> str:
    fits = [ind.fitness for ind in pop]
    total = sum(fits)
    n = len(pop)
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * 72)
    header = (
        f"{'#':>3}  {'chromosome':<{problem.encoding.length + 2}}  "
        f"{'x':<28}  {'f(x)':>12}  {'fitness':>10}  {'p_i':>7}  {'n·p_i':>7}"
    )
    lines.append(header)
    for i, ind in enumerate(pop, 1):
        p = (ind.fitness / total) if total else 0.0
        lines.append(
            f"{i:3d}  {ind.bits:<{problem.encoding.length + 2}}  "
            f"{format_x(problem, ind):<28}  {ind.objective:12.4f}  "
            f"{ind.fitness:10.4f}  {p:7.3f}  {n * p:7.2f}"
        )
    avg = sum(ind.objective for ind in pop) / n
    lines.append(f"avg f(x) = {avg:.4f}     sum fitness = {total:.4f}")
    return "\n".join(lines)


def format_history(problem: Problem, result: RunResult) -> str:
    sense = "max f" if problem.sense == "maximize" else "min f"
    lines = [f"{'gen':>4}  {sense:>12}  {'avg f':>12}  x"]
    for rec in result.history:
        best = rec.best
        lines.append(
            f"{rec.generation:4d}  {best.objective:12.4f}  {rec.average:12.4f}  "
            f"{format_x(problem, best)}"
        )
    return "\n".join(lines)


def format_best(problem: Problem, result: RunResult) -> str:
    best = result.best_ever
    lines = [
        f"Best found ({problem.sense}):  f = {best.objective:.6g}",
        f"  chromosome  {best.bits}",
        f"  {format_x(problem, best)}",
        f"  evaluations {result.evaluations}",
    ]
    if problem.optimum_f is not None:
        lines.append(f"  known optimum f = {problem.optimum_f:g}")
    return "\n".join(lines)


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    chars = []
    for v in values:
        idx = int(round((v - lo) / span * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        chars.append(blocks[idx])
    return "".join(chars)
