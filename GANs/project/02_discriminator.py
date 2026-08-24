#!/usr/bin/env python3
"""Programa 2: el discriminador, solo.

Antes del juego completo, entrenamos únicamente al DISCRIMINADOR (el detective)
para separar datos reales de un "falsificador ingenuo" fijo (una gaussiana en el
lugar equivocado). Sirve para ver que D no es más que un clasificador binario:
su acierto sube y aprende una curva de decisión D(x) ≈ P(x es real).

En un GAN de verdad, ese falsificador deja de ser fijo: es el generador, que va
mejorando. Eso es el programa 03.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gan.cli import build_parser, load          # noqa: E402
from gan.data import Problem                     # noqa: E402
from gan.gan import sigmoid                       # noqa: E402
from gan.mlp import MLP, Adam, add_grads, scale_grads, zero_grads  # noqa: E402


def main() -> None:
    args = build_parser("Entrena solo al discriminador contra un falso fijo.").parse_args()
    cfg = load(args)
    rng = random.Random(cfg.seed)
    problem = Problem(cfg, rng)

    # Falsificador ingenuo y fijo: una gaussiana desplazada respecto a la real.
    fake_mean, fake_std = cfg.low + (cfg.high - cfg.low) * 0.25, 0.6

    def fake_sample() -> float:
        return rng.gauss(fake_mean, fake_std)

    D = MLP([1] + cfg.disc_hidden + [1], rng)
    opt = Adam(D, cfg.lr_d)
    steps = args.steps if args.steps is not None else 400
    batch = cfg.batch

    print(f"Real: {problem.describe()}")
    print(f"Falso (fijo): N(media={fake_mean:.2f}, desv={fake_std:.2f})")
    print(f"Entrenando D durante {steps} pasos...\n")
    print(f"{'paso':>5} | {'pérdida D':>10} | {'acierto':>8}")
    print("-" * 30)

    for step in range(1, steps + 1):
        acc = zero_grads(D)
        loss = 0.0
        correct = 0
        for _ in range(batch):
            x = [problem.real_sample()]
            cache = {}
            p = sigmoid(D.forward(x, cache)[0])
            loss += -math.log(p + 1e-9)
            correct += 1 if p > 0.5 else 0
            grads, _ = D.backward(cache, [p - 1.0])
            add_grads(acc, grads)
            xf = [fake_sample()]
            cache = {}
            p = sigmoid(D.forward(xf, cache)[0])
            loss += -math.log(1.0 - p + 1e-9)
            correct += 1 if p < 0.5 else 0
            grads, _ = D.backward(cache, [p])
            add_grads(acc, grads)
        scale_grads(acc, 1.0 / (2 * batch))
        opt.step(acc)
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f"{step:5d} | {loss / (2 * batch):10.4f} | {correct / (2 * batch):8.2%}")

    print("\nCurva de decisión aprendida  D(x) = P(x es real):")
    print("(esperado: ~0 sobre la zona del falso, ~1 sobre la zona real)\n")
    steps_x = 29
    for i in range(steps_x):
        x = cfg.low + (cfg.high - cfg.low) * i / (steps_x - 1)
        d = sigmoid(D.forward([x])[0])
        bar = "█" * int(round(d * 40))
        print(f"x={x:5.2f}  D={d:4.2f} | {bar}")

    print("\nD aprendió a separar real de falso. En el GAN, cuando D ya no puede")
    print("distinguirlos (D≈0.5 en todas partes), el generador ha ganado.")


if __name__ == "__main__":
    main()
