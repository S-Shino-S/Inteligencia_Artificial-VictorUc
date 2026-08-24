"""The learning problem: the real 1D distribution and the noise prior.

Everything here is 1D so the result can be checked by eye. The generator will
try to turn `noise_batch()` samples into numbers that look like `real_batch()`.
"""

from __future__ import annotations

import random
from typing import List

from gan.config import Config


class Problem:
    def __init__(self, cfg: Config, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.kind = str(cfg.target.get("kind", "gaussian")).lower()
        if self.kind == "mixture":
            comps = cfg.target["components"]
            self.components = [
                (float(c["mean"]), float(c["std"]), float(c.get("weight", 1.0)))
                for c in comps
            ]
            total = sum(w for _, _, w in self.components)
            self.components = [(m, s, w / total) for m, s, w in self.components]
        else:
            self.mean = float(cfg.target.get("mean", 4.0))
            self.std = float(cfg.target.get("std", 0.6))

    # --- real data -----------------------------------------------------
    def real_sample(self) -> float:
        if self.kind == "mixture":
            r = self.rng.random()
            acc = 0.0
            for m, s, w in self.components:
                acc += w
                if r <= acc:
                    return self.rng.gauss(m, s)
            m, s, _ = self.components[-1]
            return self.rng.gauss(m, s)
        return self.rng.gauss(self.mean, self.std)

    def real_batch(self, n: int) -> List[List[float]]:
        return [[self.real_sample()] for _ in range(n)]

    # --- noise (latent space) -----------------------------------------
    def noise_vector(self) -> List[float]:
        if self.cfg.noise_kind == "gaussian":
            return [self.rng.gauss(0.0, 1.0) for _ in range(self.cfg.noise_dim)]
        return [self.rng.uniform(-1.0, 1.0) for _ in range(self.cfg.noise_dim)]

    def noise_batch(self, n: int) -> List[List[float]]:
        return [self.noise_vector() for _ in range(n)]

    def describe(self) -> str:
        if self.kind == "mixture":
            parts = ", ".join(f"N({m:.1f}, {s:.1f})×{w:.2f}" for m, s, w in self.components)
            return f"mezcla: {parts}"
        return f"gaussiana N(media={self.mean:.2f}, desv={self.std:.2f})"
