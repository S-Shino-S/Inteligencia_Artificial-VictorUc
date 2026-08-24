#!/usr/bin/env python3
"""Programa 4: generar con el modelo entrenado.

Carga el generador guardado por 03_train.py (o entrena uno al vuelo si no existe)
y produce muestras nuevas a partir de puro ruido. Compara su distribución con la
real para comprobar que G aprendió a imitarla.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gan.cli import build_parser, load          # noqa: E402
from gan.data import Problem                     # noqa: E402
from gan.gan import GAN                           # noqa: E402
from gan.io import load_model                     # noqa: E402
from gan import format as fmt                      # noqa: E402


def main() -> None:
    args = build_parser("Genera muestras con el generador entrenado.").parse_args()
    cfg = load(args)
    rng = random.Random(cfg.seed)
    problem = Problem(cfg, rng)
    gan = GAN(cfg, rng)

    if args.model.exists():
        load_model(gan, args.model)
        print(f"Modelo cargado desde {args.model.name}.")
    else:
        print(f"No hay {args.model.name}; entrenando uno rápido ({cfg.steps} pasos)...")
        gan.train(problem)

    n = cfg.eval_samples
    reals = [problem.real_sample() for _ in range(n)]
    fake = gan.generate(problem, n)

    print(f"\nObjetivo: {problem.describe()}")
    print(fmt.stat_line("REAL     ", reals))
    print(fmt.stat_line("GENERADO ", fake))

    print("\nAlgunas muestras generadas desde ruido aleatorio:")
    sample = ", ".join(f"{v:.2f}" for v in fake[:12])
    print(f"  {sample}")

    print("\nHistograma (█ real, ▓ generado):\n")
    print(fmt.compare_block(reals, fake, cfg.bins, cfg.low, cfg.high))

    print("\nCada número salió de un vector de ruido distinto pasado por G.")
    print("Cambia data/gan.yaml (p. ej. target a 'mixture') y reentrena con 03_train.py.")


if __name__ == "__main__":
    main()
