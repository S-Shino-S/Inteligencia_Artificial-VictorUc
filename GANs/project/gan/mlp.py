"""A tiny multilayer perceptron in pure Python (no NumPy).

Fully-connected layers, LeakyReLU on the hidden layers and a linear output.
Forward and backward are written by hand so every gradient is visible.

The key detail for GANs: `backward` returns *both* the parameter gradients and
the gradient with respect to the input. That input gradient is what lets the
generator learn *through* the discriminator without ever touching real data.
"""

from __future__ import annotations

import math
from typing import List, Tuple

LEAK = 0.2  # slope of LeakyReLU for x < 0

# A layer's gradient is (grad_W, grad_b).
LayerGrad = Tuple[List[List[float]], List[float]]


def leaky(x: float) -> float:
    return x if x > 0.0 else LEAK * x


def dleaky(x: float) -> float:
    return 1.0 if x > 0.0 else LEAK


class Linear:
    """One fully-connected layer: out = W @ x + b."""

    def __init__(self, n_in: int, n_out: int, rng) -> None:
        scale = math.sqrt(2.0 / n_in)  # He initialization
        self.n_in = n_in
        self.n_out = n_out
        self.W = [[rng.gauss(0.0, scale) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0 for _ in range(n_out)]

    def forward(self, x: List[float]) -> List[float]:
        out = []
        for o in range(self.n_out):
            row = self.W[o]
            acc = self.b[o]
            for i in range(self.n_in):
                acc += row[i] * x[i]
            out.append(acc)
        return out


class MLP:
    """A stack of Linear layers. Hidden layers use LeakyReLU; output is linear."""

    def __init__(self, sizes: List[int], rng) -> None:
        self.layers = [Linear(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]

    def forward(self, x: List[float], cache: dict | None = None) -> List[float]:
        """Return the output. If `cache` is given, store activations for backward."""
        acts = [x]        # input fed to each layer
        pre = []          # pre-activation (z) of each layer
        a = x
        last = len(self.layers) - 1
        for li, layer in enumerate(self.layers):
            z = layer.forward(a)
            pre.append(z)
            a = z[:] if li == last else [leaky(v) for v in z]
            acts.append(a)
        if cache is not None:
            cache["acts"] = acts
            cache["pre"] = pre
        return a

    def backward(self, cache: dict, d_out: List[float]) -> Tuple[List[LayerGrad], List[float]]:
        """Backprop a loss gradient `d_out` (wrt the output).

        Returns (per-layer gradients, gradient wrt the input).
        """
        acts = cache["acts"]
        pre = cache["pre"]
        last = len(self.layers) - 1
        grads: List[LayerGrad] = [([], []) for _ in self.layers]
        delta = d_out[:]  # dL/da_out for the current layer
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            # Move through the activation to get dL/dz.
            if li == last:
                dz = delta
            else:
                z = pre[li]
                dz = [delta[o] * dleaky(z[o]) for o in range(layer.n_out)]
            a_in = acts[li]
            gW = [[dz[o] * a_in[i] for i in range(layer.n_in)] for o in range(layer.n_out)]
            gb = list(dz)
            grads[li] = (gW, gb)
            # Propagate to the input of this layer: dL/da_in = W^T dz.
            d_in = [0.0] * layer.n_in
            for o in range(layer.n_out):
                do = dz[o]
                row = layer.W[o]
                for i in range(layer.n_in):
                    d_in[i] += do * row[i]
            delta = d_in
        return grads, delta


# --- Gradient bookkeeping (accumulate a mini-batch, then step) --------------

def zero_grads(mlp: MLP) -> List[LayerGrad]:
    return [([[0.0] * l.n_in for _ in range(l.n_out)], [0.0] * l.n_out) for l in mlp.layers]


def add_grads(dst: List[LayerGrad], src: List[LayerGrad]) -> None:
    for (dW, db), (sW, sb) in zip(dst, src):
        for o in range(len(dW)):
            drow, srow = dW[o], sW[o]
            for i in range(len(drow)):
                drow[i] += srow[i]
            db[o] += sb[o]


def scale_grads(g: List[LayerGrad], factor: float) -> None:
    for dW, db in g:
        for o in range(len(dW)):
            row = dW[o]
            for i in range(len(row)):
                row[i] *= factor
            db[o] *= factor


class Adam:
    """Adam optimizer. Plain SGD struggles with the moving GAN target, so we
    use Adam to keep the demo stable and reproducible."""

    def __init__(self, mlp: MLP, lr: float, b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.mlp = mlp
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.mW = [[[0.0] * l.n_in for _ in range(l.n_out)] for l in mlp.layers]
        self.vW = [[[0.0] * l.n_in for _ in range(l.n_out)] for l in mlp.layers]
        self.mb = [[0.0] * l.n_out for l in mlp.layers]
        self.vb = [[0.0] * l.n_out for l in mlp.layers]

    def step(self, grads: List[LayerGrad]) -> None:
        self.t += 1
        b1, b2, eps, lr = self.b1, self.b2, self.eps, self.lr
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for li, layer in enumerate(self.mlp.layers):
            gW, gb = grads[li]
            mW, vW = self.mW[li], self.vW[li]
            mb, vb = self.mb[li], self.vb[li]
            for o in range(layer.n_out):
                row = layer.W[o]
                gWo, mWo, vWo = gW[o], mW[o], vW[o]
                for i in range(layer.n_in):
                    g = gWo[i]
                    mWo[i] = b1 * mWo[i] + (1 - b1) * g
                    vWo[i] = b2 * vWo[i] + (1 - b2) * g * g
                    mhat = mWo[i] / bc1
                    vhat = vWo[i] / bc2
                    row[i] -= lr * mhat / (math.sqrt(vhat) + eps)
                g = gb[o]
                mb[o] = b1 * mb[o] + (1 - b1) * g
                vb[o] = b2 * vb[o] + (1 - b2) * g * g
                mhat = mb[o] / bc1
                vhat = vb[o] / bc2
                layer.b[o] -= lr * mhat / (math.sqrt(vhat) + eps)
