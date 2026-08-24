"""Save and load a trained generator/discriminator to plain JSON."""

from __future__ import annotations

import json
from pathlib import Path

from gan.gan import GAN
from gan.mlp import MLP


def _dump_mlp(mlp: MLP) -> list:
    return [{"W": layer.W, "b": layer.b} for layer in mlp.layers]


def _load_into(mlp: MLP, blob: list) -> None:
    if len(blob) != len(mlp.layers):
        raise ValueError("saved network shape does not match the current config")
    for layer, saved in zip(mlp.layers, blob):
        layer.W = [list(row) for row in saved["W"]]
        layer.b = list(saved["b"])


def save_model(gan: GAN, path: str | Path) -> None:
    data = {
        "noise_dim": gan.cfg.noise_dim,
        "generator": _dump_mlp(gan.G),
        "discriminator": _dump_mlp(gan.D),
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def load_model(gan: GAN, path: str | Path) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _load_into(gan.G, data["generator"])
    _load_into(gan.D, data["discriminator"])
