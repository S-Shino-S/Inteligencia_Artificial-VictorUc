"""Simple generational GA (Holland / Goldberg)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ga.encoding import Encoding
from ga.fitness import is_better, selection_fitness
from ga.operators import crossover_at, mutate, one_point_crossover, roulette, tournament
from ga.problem import Problem, Replay


@dataclass
class Individual:
    bits: str
    x: tuple[float, ...]
    objective: float
    fitness: float = 0.0


@dataclass
class StepEvent:
    message: str


@dataclass
class GenerationRecord:
    generation: int
    best: Individual
    average: float
    worst: Individual
    population: list[Individual]


@dataclass
class RunResult:
    history: list[GenerationRecord]
    best_ever: Individual
    evaluations: int
    events: list[StepEvent] = field(default_factory=list)


def evaluate(bits: str, encoding: Encoding, objective) -> Individual:
    xs = encoding.decode(bits)
    return Individual(bits=bits, x=xs, objective=float(objective(xs)))


def random_bits(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(length))


def seed_population(problem: Problem, rng: random.Random) -> list[Individual]:
    n = problem.population
    if problem.initial:
        bits_list = list(problem.initial)
        if len(bits_list) != n:
            raise ValueError(f"initial population has {len(bits_list)} strings, expected {n}")
    else:
        bits_list = [random_bits(problem.encoding.length, rng) for _ in range(n)]
    pop = [evaluate(b, problem.encoding, problem.objective) for b in bits_list]
    _assign_fitness(pop, problem.sense)
    return pop


def _assign_fitness(pop: list[Individual], sense: str) -> None:
    fits = selection_fitness([ind.objective for ind in pop], sense)
    for ind, fit in zip(pop, fits):
        ind.fitness = fit


def _pick(pop: list[Individual], problem: Problem, rng: random.Random) -> int:
    fits = [ind.fitness for ind in pop]
    if problem.selection == "tournament":
        return tournament(fits, rng, problem.tournament_k)
    if problem.selection == "roulette":
        return roulette(fits, rng)
    raise ValueError(f"unknown selection {problem.selection!r}")


def _best(pop: list[Individual], sense: str) -> Individual:
    winner = pop[0]
    for ind in pop[1:]:
        if is_better(ind.objective, winner.objective, sense):
            winner = ind
    return winner


def _worst(pop: list[Individual], sense: str) -> Individual:
    loser = pop[0]
    for ind in pop[1:]:
        if is_better(loser.objective, ind.objective, sense):
            loser = ind
    return loser


def _record(generation: int, pop: list[Individual], sense: str) -> GenerationRecord:
    return GenerationRecord(
        generation=generation,
        best=_best(pop, sense),
        average=sum(ind.objective for ind in pop) / len(pop),
        worst=_worst(pop, sense),
        population=list(pop),
    )


def next_generation(
    pop: list[Individual],
    problem: Problem,
    rng: random.Random,
    events: list[StepEvent] | None = None,
    replay: Replay | None = None,
) -> list[Individual]:
    """Build P' from P (generational replacement, optional elitism)."""
    n = len(pop)
    log = events.append if events is not None else lambda _m: None
    children_bits: list[str] = []

    elites = []
    if problem.elitism > 0 and replay is None:
        ranked = sorted(
            pop,
            key=lambda ind: ind.objective,
            reverse=(problem.sense == "maximize"),
        )
        elites = [ind.bits for ind in ranked[: problem.elitism]]
        children_bits.extend(elites)
        log(StepEvent(f"elitism: keep {len(elites)} best chromosome(s)"))

    if replay is not None:
        pool = list(replay.mating_pool)
        log(StepEvent("mating pool (lecture replay): " + ", ".join(pool)))
        pairs = list(zip(pool[0::2], pool[1::2], replay.cuts))
        for a, b, cut in pairs:
            c1, c2 = crossover_at(a, b, cut)
            log(StepEvent(f"crossover cut={cut}:  {a} × {b}  →  {c1} , {c2}"))
            children_bits.extend([c1, c2])
        children_bits = children_bits[:n]
    else:
        while len(children_bits) < n:
            i = _pick(pop, problem, rng)
            j = _pick(pop, problem, rng)
            a, b = pop[i].bits, pop[j].bits
            c1, c2, cut = one_point_crossover(a, b, rng, problem.p_crossover)
            if cut is None:
                log(StepEvent(f"no crossover:  {a} , {b}"))
            else:
                log(StepEvent(f"crossover cut={cut}:  {a} × {b}  →  {c1} , {c2}"))
            children_bits.append(c1)
            if len(children_bits) < n:
                children_bits.append(c2)

    next_pop = []
    for k, bits in enumerate(children_bits):
        if replay is not None:
            new_bits, flipped = bits, []
        else:
            # Do not mutate elites copied at the front.
            if k < len(elites):
                new_bits, flipped = bits, []
            else:
                new_bits, flipped = mutate(bits, rng, problem.p_mutation)
        if flipped:
            log(StepEvent(f"mutate {bits}  →  {new_bits}  (bits {flipped})"))
        next_pop.append(evaluate(new_bits, problem.encoding, problem.objective))
    _assign_fitness(next_pop, problem.sense)
    return next_pop


def run_ga(
    problem: Problem,
    rng: random.Random | None = None,
    generations: int | None = None,
    verbose: bool = False,
    use_replay: bool = False,
) -> RunResult:
    rng = rng or random.Random(problem.seed)
    gens = problem.generations if generations is None else generations
    events: list[StepEvent] = []
    pop = seed_population(problem, rng)
    history = [_record(0, pop, problem.sense)]
    best_ever = history[0].best
    evaluations = len(pop)

    for g in range(1, gens + 1):
        replay = problem.replay if (use_replay and g == 1) else None
        pop = next_generation(
            pop,
            problem,
            rng,
            events=events if verbose else None,
            replay=replay,
        )
        rec = _record(g, pop, problem.sense)
        history.append(rec)
        evaluations += len(pop)
        if is_better(rec.best.objective, best_ever.objective, problem.sense):
            best_ever = rec.best

    return RunResult(history=history, best_ever=best_ever, evaluations=evaluations, events=events)


def random_search(problem: Problem, evaluations: int, rng: random.Random) -> Individual:
    """Same encoding and f, but sample chromosomes uniformly (no selection)."""
    best = evaluate(random_bits(problem.encoding.length, rng), problem.encoding, problem.objective)
    for _ in range(evaluations - 1):
        ind = evaluate(random_bits(problem.encoding.length, rng), problem.encoding, problem.objective)
        if is_better(ind.objective, best.objective, problem.sense):
            best = ind
    return best
