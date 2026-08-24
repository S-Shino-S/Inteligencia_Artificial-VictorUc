"""A GAN whose data lives in several dimensions (used by program 05).

Same minimax game and same code as `gan.gan`, but the generator emits a vector
and the discriminator reads a vector. The MLP in `gan.mlp` is already
dimension-agnostic, so the only real change is standardizing per axis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List

from gan.config import Config
from gan.data2d import Target2D
from gan.gan import sigmoid
from gan.mlp import MLP, Adam, add_grads, scale_grads, zero_grads


@dataclass
class Report2D:
    step: int
    loss_d: float
    loss_g: float
    acc_d: float


class GAN2D:
    def __init__(self, cfg: Config, target: Target2D, rng: random.Random,
                 data_dim: int = 2) -> None:
        self.cfg = cfg
        self.rng = rng
        self.dim = data_dim
        self.G = MLP([cfg.noise_dim] + cfg.gen_hidden + [data_dim], rng)
        self.D = MLP([data_dim] + cfg.disc_hidden + [1], rng)
        self.opt_g = Adam(self.G, cfg.lr_g)
        self.opt_d = Adam(self.D, cfg.lr_d)
        self.center = (target.low + target.high) / 2.0
        self.spread = (target.high - target.low) / 4.0

    def _std(self, v: List[float]) -> List[float]:
        return [(c - self.center) / self.spread for c in v]

    def _destd(self, u: List[float]) -> List[float]:
        return [c * self.spread + self.center for c in u]

    def noise(self) -> List[float]:
        if self.cfg.noise_kind == "gaussian":
            return [self.rng.gauss(0.0, 1.0) for _ in range(self.cfg.noise_dim)]
        return [self.rng.uniform(-1.0, 1.0) for _ in range(self.cfg.noise_dim)]

    def generate(self, n: int) -> List[List[float]]:
        return [self._destd(self.G.forward(self.noise())) for _ in range(n)]

    def step_d(self, target: Target2D, batch: int) -> tuple[float, float]:
        acc = zero_grads(self.D)
        loss = 0.0
        correct = 0
        reals = [self._std(v) for v in target.batch(batch)]
        fakes = [self.G.forward(self.noise()) for _ in range(batch)]
        for x in reals:
            cache = {}
            p = sigmoid(self.D.forward(x, cache)[0])
            loss += -math.log(p + 1e-9)
            correct += 1 if p > 0.5 else 0
            grads, _ = self.D.backward(cache, [p - 1.0])
            add_grads(acc, grads)
        for fx in fakes:
            cache = {}
            p = sigmoid(self.D.forward(fx, cache)[0])
            loss += -math.log(1.0 - p + 1e-9)
            correct += 1 if p < 0.5 else 0
            grads, _ = self.D.backward(cache, [p])
            add_grads(acc, grads)
        scale_grads(acc, 1.0 / (2 * batch))
        self.opt_d.step(acc)
        return loss / (2 * batch), correct / (2 * batch)

    def step_g(self, batch: int) -> float:
        acc = zero_grads(self.G)
        loss = 0.0
        for _ in range(batch):
            cg = {}
            fake = self.G.forward(self.noise(), cg)
            cd = {}
            p = sigmoid(self.D.forward(fake, cd)[0])
            loss += -math.log(p + 1e-9)
            _, d_fake = self.D.backward(cd, [p - 1.0])
            grads, _ = self.G.backward(cg, d_fake)
            add_grads(acc, grads)
        scale_grads(acc, 1.0 / batch)
        self.opt_g.step(acc)
        return loss / batch

    def train(self, target: Target2D,
              on_report: Callable[[Report2D], None] | None = None) -> List[Report2D]:
        cfg = self.cfg
        history: List[Report2D] = []
        for step in range(1, cfg.steps + 1):
            loss_d = acc_d = 0.0
            for _ in range(cfg.d_steps):
                loss_d, acc_d = self.step_d(target, cfg.batch)
            loss_g = self.step_g(cfg.batch)
            if step == 1 or step % cfg.report_every == 0 or step == cfg.steps:
                rep = Report2D(step, loss_d, loss_g, acc_d)
                history.append(rep)
                if on_report:
                    on_report(rep)
        return history
