#!/usr/bin/env python3
"""Programa 3: el juego completo (entrenamiento adversario).

Generador y discriminador se entrenan a la vez:
  - D aprende a separar reales de G(z).
  - G aprende a producir G(z) que engañen a D.

Cada cierto número de pasos imprimimos las pérdidas, el acierto de D (baja hacia
~50% cuando G mejora) y un histograma REAL vs GENERADO para ver la convergencia.
Al final guarda el modelo en model.json para el programa 04.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gan.cli import build_parser, load          # noqa: E402
from gan.data import Problem                     # noqa: E402
from gan.gan import GAN, Report                   # noqa: E402
from gan.io import save_model                     # noqa: E402
from gan import format as fmt                      # noqa: E402


def main() -> None:
    args = build_parser("Entrena el GAN completo (minimax).").parse_args()
    cfg = load(args)
    rng = random.Random(cfg.seed)
    problem = Problem(cfg, rng)
    gan = GAN(cfg, rng)

    reals = [problem.real_sample() for _ in range(cfg.eval_samples)]

    print(f"Objetivo: {problem.describe()}")
    print(f"G: {cfg.noise_dim}→{'→'.join(map(str, cfg.gen_hidden))}→1   "
          f"D: 1→{'→'.join(map(str, cfg.disc_hidden))}→1")
    print(f"Pasos={cfg.steps}  lote={cfg.batch}  lr_g={cfg.lr_g}  lr_d={cfg.lr_d}\n")
    print(f"{'paso':>6} | {'pérdida D':>10} | {'pérdida G':>10} | {'acierto D':>9} | gen media/desv")
    print("-" * 68)

    def report(r: Report) -> None:
        print(f"{r.step:6d} | {r.loss_d:10.4f} | {r.loss_g:10.4f} | "
              f"{r.acc_d:9.2%} | {r.fake_mean:6.3f} / {r.fake_std:5.3f}")

    gan.train(problem, on_report=report)

    fake = gan.generate(problem, cfg.eval_samples)
    print("\n" + fmt.stat_line("REAL     ", reals))
    print(fmt.stat_line("GENERADO ", fake))
    print("\nHistograma final (█ real, ▓ generado):\n")
    print(fmt.compare_block(reals, fake, cfg.bins, cfg.low, cfg.high))

    save_model(gan, args.model)
    print(f"\nModelo guardado en {args.model.name}. Genera muestras con 04_generate.py.")


if __name__ == "__main__":
    main()
