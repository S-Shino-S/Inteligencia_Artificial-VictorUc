#!/usr/bin/env python3
"""Programa 1: el problema.

Antes de entrenar nada, miramos las dos piezas del juego:
  - la distribución REAL que queremos imitar (target), y
  - el RUIDO z que el generador recibirá como entrada.

El generador tendrá que aprender a convertir ese ruido sin forma en números
que sigan la distribución real. Aquí solo los dibujamos.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gan.cli import build_parser, load          # noqa: E402
from gan.data import Problem                     # noqa: E402
from gan import format as fmt                     # noqa: E402


def main() -> None:
    args = build_parser("Muestra la distribución real y el ruido de entrada.").parse_args()
    cfg = load(args)
    rng = random.Random(cfg.seed)
    problem = Problem(cfg, rng)

    n = cfg.eval_samples
    reals = [problem.real_sample() for _ in range(n)]

    print(f"Distribución real objetivo: {problem.describe()}")
    print(f"{fmt.stat_line('  reales', reals)}\n")
    print(fmt.histogram_block(reals, cfg.bins, cfg.low, cfg.high,
                              label="Histograma de datos REALES (lo que queremos generar):"))

    print("\nRuido de entrada del generador (espacio latente z):")
    dim = cfg.noise_dim
    zs = [problem.noise_vector() for _ in range(n)]
    first_dim = [z[0] for z in zs]
    print(f"  z ∈ R^{dim}, tipo '{cfg.noise_kind}'. Cada muestra es un vector de {dim} números.")
    print(f"  Ejemplo: z = [{', '.join(f'{v:+.2f}' for v in zs[0])}]")
    print()
    print(fmt.histogram_block(first_dim, cfg.bins, cfg.low, cfg.high,
                              char="·",
                              label="Primera dimensión del ruido (aún NO se parece a los datos):"))

    print("\nLa tarea del GAN: aprender una función G que transforme el ruido de")
    print("arriba en la distribución real de más arriba. Corre 03_train.py para verlo.")


if __name__ == "__main__":
    main()
