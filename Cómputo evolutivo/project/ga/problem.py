"""YAML problem: function, encoding, GA parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ga.encoding import Encoding, Gene
from ga.functions import Objective, get_function


@dataclass(frozen=True)
class Replay:
    """Optional scripted generation 0 → 1 (Goldberg lecture numbers)."""

    mating_pool: tuple[str, ...]
    cuts: tuple[int, ...]


@dataclass(frozen=True)
class Problem:
    name: str
    sense: str
    function_name: str
    objective: Objective
    encoding: Encoding
    population: int
    generations: int
    p_crossover: float
    p_mutation: float
    selection: str
    tournament_k: int
    elitism: int
    seed: int | None
    initial: tuple[str, ...] | None
    replay: Replay | None
    optimum_x: tuple[float, ...] | None
    optimum_f: float | None
    source: Path | None = None

    @property
    def n(self) -> int:
        return self.population


def load_problem(path: str | Path) -> Problem:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_problem(raw, source=path)


def parse_problem(raw: dict[str, Any], source: Path | None = None) -> Problem:
    sense = str(raw.get("sense") or "maximize").lower()
    if sense not in {"maximize", "minimize"}:
        raise ValueError("sense must be maximize or minimize")
    fn_name = str(raw.get("function") or "x_squared")
    genes = []
    for item in raw.get("variables") or []:
        genes.append(
            Gene(
                gid=str(item["id"]),
                bits=int(item["bits"]),
                low=float(item.get("low", 0)),
                high=float(item.get("high", (1 << int(item["bits"])) - 1)),
                kind=str(item.get("kind") or "integer"),
            )
        )
    if not genes:
        raise ValueError("variables: needs at least one gene")
    encoding = Encoding(tuple(genes))

    initial = raw.get("initial")
    initial_t = tuple(str(s).strip() for s in initial) if initial else None
    if initial_t:
        for chrom in initial_t:
            encoding.decode(chrom)

    replay = None
    raw_replay = raw.get("replay")
    if raw_replay:
        pool = tuple(str(s).strip() for s in raw_replay["mating_pool"])
        cuts = tuple(int(c) for c in raw_replay["cuts"])
        replay = Replay(pool, cuts)

    opt = raw.get("optimum") or {}
    opt_x = tuple(float(v) for v in opt["x"]) if opt.get("x") is not None else None
    opt_f = float(opt["f"]) if opt.get("f") is not None else None

    pop = int(raw.get("population") or 4)
    if pop < 2:
        raise ValueError("population must be ≥ 2")

    return Problem(
        name=str(raw.get("name") or "GA problem"),
        sense=sense,
        function_name=fn_name,
        objective=get_function(fn_name),
        encoding=encoding,
        population=pop,
        generations=int(raw.get("generations") or 1),
        p_crossover=float(raw.get("p_crossover") if raw.get("p_crossover") is not None else 0.8),
        p_mutation=float(raw.get("p_mutation") if raw.get("p_mutation") is not None else 0.01),
        selection=str(raw.get("selection") or "roulette"),
        tournament_k=int(raw.get("tournament_k") or 2),
        elitism=int(raw.get("elitism") or 0),
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        initial=initial_t,
        replay=replay,
        optimum_x=opt_x,
        optimum_f=opt_f,
        source=source,
    )


def format_problem(problem: Problem) -> str:
    enc = problem.encoding
    lines = [problem.name, ""]
    lines.append(f"Sense:       {problem.sense}  f  =  {problem.function_name}")
    lines.append(f"Population:  {problem.population}")
    lines.append(f"Generations: {problem.generations}")
    lines.append(f"Selection:   {problem.selection}" + (f" (k={problem.tournament_k})" if problem.selection == "tournament" else ""))
    lines.append(f"p_crossover: {problem.p_crossover}")
    lines.append(f"p_mutation:  {problem.p_mutation}   (per bit)")
    lines.append(f"Elitism:     {problem.elitism}")
    if problem.seed is not None:
        lines.append(f"Seed:        {problem.seed}")
    lines.append("")
    lines.append("Encoding  (chromosome length = {} bits):".format(enc.length))
    for g in enc.genes:
        kind = "integer" if g.kind == "integer" else "real"
        lines.append(f"  {g.gid}: {g.bits} bits, {kind} in [{g.low:g}, {g.high:g}]")
    if problem.optimum_f is not None:
        xs = problem.encoding.format_x(problem.optimum_x) if problem.optimum_x else "?"
        lines.append("")
        lines.append(f"Known optimum (for checking):  {xs}   f = {problem.optimum_f:g}")
    if problem.initial:
        lines.append("")
        lines.append("Fixed initial population:")
        for bits in problem.initial:
            xs = enc.decode(bits)
            f = problem.objective(xs)
            lines.append(f"  {bits}   {enc.format_x(xs)}   f={f:g}")
    return "\n".join(lines)
