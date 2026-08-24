#!/usr/bin/env python3
"""Programa 5: un objetivo en 2D con dispersión ASCII.

El mismo GAN, pero ahora los datos son puntos (x, y). Con un objetivo en forma de
ANILLO se ve muy bien la estructura: si el generador la captura, sus puntos (*)
cubren todo el círculo de puntos reales (o); si sufre COLAPSO DE MODOS, se
amontonan en un trozo del anillo y dejan huecos.

Mira el gráfico "antes" (ruido sin forma) y el "después" (tras entrenar).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gan.cli import build_parser, load          # noqa: E402
from gan.data2d import Target2D                  # noqa: E402
from gan.gan2d import GAN2D, Report2D            # noqa: E402
from gan import format as fmt                     # noqa: E402


def ring_metrics(points, radius):
    """Radio medio y cobertura angular (fracción de sectores del anillo con
    al menos un punto): una cobertura baja delata colapso de modos."""
    sectors = 24
    hit = set()
    radii = []
    for x, y in points:
        radii.append(math.hypot(x, y))
        ang = math.atan2(y, x)
        idx = int((ang + math.pi) / (2 * math.pi) * sectors) % sectors
        hit.add(idx)
    mean_r = sum(radii) / len(radii) if radii else 0.0
    return mean_r, len(hit) / sectors


def blob_metrics(points, centers):
    counts = [0] * len(centers)
    for x, y in points:
        best, bi = 1e18, 0
        for i, (cx, cy) in enumerate(centers):
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < best:
                best, bi = d, i
        counts[bi] += 1
    covered = sum(1 for c in counts if c >= 0.02 * len(points))
    return covered, len(centers)


def main() -> None:
    args = build_parser("GAN con un objetivo 2D y dispersión ASCII.").parse_args()
    cfg = load(args)
    rng = random.Random(cfg.seed)
    target = Target2D(cfg.two_d, rng)
    gan = GAN2D(cfg, target, rng)

    n_plot = min(cfg.eval_samples, 900)
    reals = target.batch(cfg.eval_samples)

    print(f"Objetivo 2D: {target.describe()}   ventana [{target.low}, {target.high}]²")
    print(f"G: {cfg.noise_dim}→{'→'.join(map(str, cfg.gen_hidden))}→2   "
          f"D: 2→{'→'.join(map(str, cfg.disc_hidden))}→1")
    print(f"Pasos={cfg.steps}  lote={cfg.batch}  d_steps={cfg.d_steps}\n")

    print("ANTES de entrenar (el generador es puro ruido):\n")
    print(fmt.scatter_block(reals[:n_plot], gan.generate(n_plot),
                            target.low, target.high))

    print(f"\n{'paso':>6} | {'pérdida D':>10} | {'pérdida G':>10} | {'acierto D':>9} | estructura")
    print("-" * 66)

    def report(r: Report2D) -> None:
        gen = gan.generate(600)
        if target.kind == "blobs":
            cov, tot = blob_metrics(gen, target.centers)
            extra = f"cúmulos cubiertos {cov}/{tot}"
        else:
            mr, cov = ring_metrics(gen, target.radius)
            extra = f"radio≈{mr:4.2f}  cobertura {cov:5.0%}"
        print(f"{r.step:6d} | {r.loss_d:10.4f} | {r.loss_g:10.4f} | "
              f"{r.acc_d:9.2%} | {extra}")

    gan.train(target, on_report=report)

    fake = gan.generate(cfg.eval_samples)
    print("\nDESPUÉS de entrenar:\n")
    print(fmt.scatter_block(reals[:n_plot], fake[:n_plot],
                            target.low, target.high))

    if target.kind == "blobs":
        cov, tot = blob_metrics(fake, target.centers)
        print(f"\nCúmulos cubiertos: {cov}/{tot} "
              f"({'todos' if cov == tot else 'faltan modos → colapso'}).")
    else:
        mr, cov = ring_metrics(fake, target.radius)
        print(f"\nRadio medio generado: {mr:.2f} (objetivo {target.radius:.1f}).  "
              f"Cobertura del anillo: {cov:.0%}.")
    print("Sube --steps o cambia data/gan.yaml (target2d) para experimentar.")


if __name__ == "__main__":
    main()
