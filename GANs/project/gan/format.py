"""ASCII rendering for the 1D distributions."""

from __future__ import annotations

from typing import List

from gan import metrics


def _bar(count: int, scale: float, char: str) -> str:
    return char * int(round(count * scale))


def histogram_block(values: List[float], bins: int, low: float, high: float,
                    width: int = 42, char: str = "█", label: str = "") -> str:
    counts = metrics.histogram(values, bins, low, high)
    peak = max(counts) or 1
    scale = width / peak
    step = (high - low) / bins
    lines = []
    if label:
        lines.append(label)
    for i, c in enumerate(counts):
        center = low + (i + 0.5) * step
        lines.append(f"{center:5.2f} | {_bar(c, scale, char):<{width}} {c}")
    return "\n".join(lines)


def compare_block(real: List[float], fake: List[float], bins: int, low: float,
                  high: float, width: int = 22) -> str:
    """Two histograms side by side: real (izq) vs generado (der)."""
    rc = metrics.histogram(real, bins, low, high)
    fc = metrics.histogram(fake, bins, low, high)
    peak = max(max(rc), max(fc)) or 1
    scale = width / peak
    step = (high - low) / bins
    header = f"{'valor':>6}  {'REAL':>{width}}   {'GENERADO':<{width}}"
    lines = [header, "-" * len(header)]
    for i in range(bins):
        center = low + (i + 0.5) * step
        rbar = ("█" * int(round(rc[i] * scale))).rjust(width)
        fbar = ("▓" * int(round(fc[i] * scale))).ljust(width)
        lines.append(f"{center:6.2f}  {rbar} | {fbar}")
    return "\n".join(lines)


def stat_line(name: str, values: List[float]) -> str:
    return f"{name}: media={metrics.mean(values):6.3f}  desv={metrics.std(values):6.3f}"


def scatter_block(reals, fakes, low: float, high: float,
                  width: int = 58, height: int = 24) -> str:
    """Overlay two point clouds on an ASCII grid.

    'o' = real, '*' = generado, '@' = ambos caen en la misma celda.
    (La celda de texto es ~2× más alta que ancha, por eso width ≈ 2·height
    para que una ventana cuadrada se vea cuadrada.)
    """
    grid = [[" "] * width for _ in range(height)]
    span = high - low

    def plot(points, ch):
        for x, y in points:
            if x < low or x > high or y < low or y > high:
                continue
            col = int((x - low) / span * (width - 1))
            row = int((high - y) / span * (height - 1))
            cur = grid[row][col]
            grid[row][col] = "@" if (cur not in (" ", ch)) else ch

    plot(reals, "o")
    plot(fakes, "*")

    top = "+" + "-" * width + "+"
    lines = [top]
    for r in grid:
        lines.append("|" + "".join(r) + "|")
    lines.append(top)
    lines.append("  o = real    * = generado    @ = ambos")
    return "\n".join(lines)
