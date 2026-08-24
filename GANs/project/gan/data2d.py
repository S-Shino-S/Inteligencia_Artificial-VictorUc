"""2D target distributions for program 05.

Two shapes are enough to see the interesting behaviour:
  - `ring`  : points on a circle. Mode collapse shows up as gaps in the ring.
  - `blobs` : a few gaussian clusters. Collapse means G covers only some of them.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

Point = Tuple[float, float]


class Target2D:
    def __init__(self, spec: dict, rng: random.Random) -> None:
        self.rng = rng
        self.kind = str(spec.get("kind", "ring")).lower()
        self.radius = float(spec.get("radius", 2.0))
        self.noise_std = float(spec.get("noise_std", 0.12))
        window = spec.get("window", [-3.2, 3.2])
        self.low = float(window[0])
        self.high = float(window[1])
        self.centers = [(float(c[0]), float(c[1]))
                        for c in spec.get("centers",
                                          [[-1.5, -1.5], [1.5, 1.5],
                                           [-1.5, 1.5], [1.5, -1.5]])]
        self.blob_std = float(spec.get("blob_std", 0.22))

    def sample(self) -> Point:
        if self.kind == "blobs":
            cx, cy = self.rng.choice(self.centers)
            return (self.rng.gauss(cx, self.blob_std), self.rng.gauss(cy, self.blob_std))
        # ring (default)
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        r = self.radius + self.rng.gauss(0.0, self.noise_std)
        return (r * math.cos(angle), r * math.sin(angle))

    def batch(self, n: int) -> List[List[float]]:
        return [list(self.sample()) for _ in range(n)]

    def describe(self) -> str:
        if self.kind == "blobs":
            return f"{len(self.centers)} cúmulos gaussianos (desv={self.blob_std:.2f})"
        return f"anillo de radio {self.radius:.1f} (grosor {self.noise_std:.2f})"
