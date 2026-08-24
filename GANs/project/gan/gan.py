"""The GAN itself: a generator and a discriminator playing the minimax game.

    min_G  max_D   E_x[ log D(x) ] + E_z[ log(1 - D(G(z))) ]

Both networks are plain MLPs from `gan.mlp`. The discriminator outputs a single
logit; we apply the sigmoid inside the loss. The generator is trained with the
non-saturating loss (maximize log D(G(z))) so it still gets a strong gradient
while it is bad.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List

from gan.config import Config
from gan.data import Problem
from gan.mlp import MLP, Adam, add_grads, scale_grads, zero_grads


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Report:
    step: int
    loss_d: float
    loss_g: float
    acc_d: float
    fake_mean: float
    fake_std: float


class GAN:
    def __init__(self, cfg: Config, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.G = MLP([cfg.noise_dim] + cfg.gen_hidden + [1], rng)
        self.D = MLP([1] + cfg.disc_hidden + [1], rng)
        self.opt_g = Adam(self.G, cfg.lr_g)
        self.opt_d = Adam(self.D, cfg.lr_d)
        # Both networks work in a standardized space (~zero-mean, ~unit-scale),
        # which keeps the sigmoid and the gradients well conditioned. We only
        # convert back to real units when we hand samples to the user.
        self.center = (cfg.low + cfg.high) / 2.0
        self.spread = (cfg.high - cfg.low) / 4.0

    def _std(self, x: float) -> float:
        return (x - self.center) / self.spread

    def _destd(self, u: float) -> float:
        return u * self.spread + self.center

    # --- inference -----------------------------------------------------
    def generate(self, problem: Problem, n: int) -> List[float]:
        return [self._destd(self.G.forward(problem.noise_vector())[0]) for _ in range(n)]

    # --- one discriminator update -------------------------------------
    def step_d(self, problem: Problem, batch: int) -> tuple[float, float]:
        acc = zero_grads(self.D)
        loss = 0.0
        correct = 0
        reals = [[self._std(v[0])] for v in problem.real_batch(batch)]
        fakes = [self.G.forward(problem.noise_vector()) for _ in range(batch)]
        for x in reals:
            cache = {}
            logit = self.D.forward(x, cache)[0]
            p = sigmoid(logit)
            loss += -math.log(p + 1e-9)          # label = 1 (real)
            correct += 1 if p > 0.5 else 0
            grads, _ = self.D.backward(cache, [p - 1.0])
            add_grads(acc, grads)
        for fx in fakes:
            cache = {}
            logit = self.D.forward([fx[0]], cache)[0]
            p = sigmoid(logit)
            loss += -math.log(1.0 - p + 1e-9)    # label = 0 (fake)
            correct += 1 if p < 0.5 else 0
            grads, _ = self.D.backward(cache, [p])
            add_grads(acc, grads)
        scale_grads(acc, 1.0 / (2 * batch))
        self.opt_d.step(acc)
        return loss / (2 * batch), correct / (2 * batch)

    # --- one generator update -----------------------------------------
    def step_g(self, problem: Problem, batch: int) -> float:
        acc = zero_grads(self.G)
        loss = 0.0
        for _ in range(batch):
            cg = {}
            fake = self.G.forward(problem.noise_vector(), cg)
            cd = {}
            logit = self.D.forward([fake[0]], cd)[0]
            p = sigmoid(logit)
            loss += -math.log(p + 1e-9)          # non-saturating: pretend label 1
            # Backprop through D (no update) to get dL/d(fake), then into G.
            _, d_fake = self.D.backward(cd, [p - 1.0])
            grads, _ = self.G.backward(cg, d_fake)
            add_grads(acc, grads)
        scale_grads(acc, 1.0 / batch)
        self.opt_g.step(acc)
        return loss / batch

    # --- full training loop -------------------------------------------
    def train(self, problem: Problem, on_report: Callable[[Report], None] | None = None
              ) -> List[Report]:
        cfg = self.cfg
        history: List[Report] = []
        for step in range(1, cfg.steps + 1):
            loss_d = acc_d = 0.0
            for _ in range(cfg.d_steps):
                loss_d, acc_d = self.step_d(problem, cfg.batch)
            loss_g = self.step_g(problem, cfg.batch)
            if step == 1 or step % cfg.report_every == 0 or step == cfg.steps:
                sample = self.generate(problem, cfg.eval_samples)
                from gan import metrics
                rep = Report(step, loss_d, loss_g, acc_d,
                             metrics.mean(sample), metrics.std(sample))
                history.append(rep)
                if on_report:
                    on_report(rep)
        return history
