"""Argument parsing and config loading shared by the programs."""

from __future__ import annotations

import argparse
from pathlib import Path

from gan.config import Config, load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "data" / "gan.yaml"
DEFAULT_MODEL = ROOT / "model.json"


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML de configuración")
    p.add_argument("--steps", type=int, default=None, help="Pasos de entrenamiento")
    p.add_argument("--batch", type=int, default=None, help="Tamaño de mini-lote")
    p.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    p.add_argument("--samples", type=int, default=None, dest="samples",
                   help="Muestras a generar/evaluar")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help="Ruta del modelo (guardar/cargar)")
    p.add_argument("--quiet", action="store_true", help="Menos texto por paso")
    return p


def load(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch is not None:
        cfg.batch = args.batch
    if args.seed is not None:
        cfg.seed = args.seed
    if args.samples is not None:
        cfg.eval_samples = args.samples
    return cfg
