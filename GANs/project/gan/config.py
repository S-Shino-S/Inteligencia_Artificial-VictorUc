"""Load and validate the YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class Config:
    seed: int = 7
    target: dict = field(default_factory=dict)
    noise_dim: int = 4
    noise_kind: str = "uniform"
    gen_hidden: List[int] = field(default_factory=lambda: [16, 16])
    disc_hidden: List[int] = field(default_factory=lambda: [16, 16])
    steps: int = 2000
    batch: int = 32
    lr_g: float = 0.002
    lr_d: float = 0.002
    d_steps: int = 1
    report_every: int = 250
    bins: int = 25
    low: float = 0.0
    high: float = 8.0
    eval_samples: int = 2000
    two_d: dict = field(default_factory=dict)
    source: Path | None = None


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_config(raw, source=path)


def parse_config(raw: dict[str, Any], source: Path | None = None) -> Config:
    noise = raw.get("noise") or {}
    gen = raw.get("generator") or {}
    disc = raw.get("discriminator") or {}
    training = raw.get("training") or {}
    hist = raw.get("histogram") or {}

    cfg = Config(
        seed=int(raw.get("seed", 7)),
        target=raw.get("target") or {"kind": "gaussian", "mean": 4.0, "std": 0.6},
        noise_dim=int(noise.get("dim", 4)),
        noise_kind=str(noise.get("kind", "uniform")).lower(),
        gen_hidden=[int(h) for h in gen.get("hidden", [16, 16])],
        disc_hidden=[int(h) for h in disc.get("hidden", [16, 16])],
        steps=int(training.get("steps", 2000)),
        batch=int(training.get("batch", 32)),
        lr_g=float(training.get("lr_g", 0.002)),
        lr_d=float(training.get("lr_d", 0.002)),
        d_steps=int(training.get("d_steps", 1)),
        report_every=int(training.get("report_every", 250)),
        bins=int(hist.get("bins", 25)),
        low=float(hist.get("low", 0.0)),
        high=float(hist.get("high", 8.0)),
        eval_samples=int(raw.get("eval_samples", 2000)),
        two_d=raw.get("target2d") or {"kind": "ring", "radius": 2.0,
                                      "noise_std": 0.12, "window": [-3.2, 3.2]},
        source=source,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.noise_dim < 1:
        raise ValueError("noise.dim must be >= 1")
    if cfg.noise_kind not in ("uniform", "gaussian"):
        raise ValueError("noise.kind must be 'uniform' or 'gaussian'")
    if cfg.batch < 1 or cfg.steps < 1:
        raise ValueError("training.batch and training.steps must be >= 1")
    if cfg.high <= cfg.low:
        raise ValueError("histogram.high must be greater than histogram.low")
    kind = str(cfg.target.get("kind", "gaussian")).lower()
    if kind not in ("gaussian", "mixture"):
        raise ValueError("target.kind must be 'gaussian' or 'mixture'")
    if kind == "mixture" and not cfg.target.get("components"):
        raise ValueError("target.kind=mixture needs a non-empty 'components' list")
