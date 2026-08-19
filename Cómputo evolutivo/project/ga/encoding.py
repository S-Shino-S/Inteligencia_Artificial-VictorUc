"""Binary genotype ↔ phenotype (integer or real variables)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gene:
    gid: str
    bits: int
    low: float
    high: float
    kind: str  # "integer" | "real"

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError(f"{self.gid}: bits must be ≥ 1")
        if self.kind not in {"integer", "real"}:
            raise ValueError(f"{self.gid}: kind must be integer or real")
        if self.high < self.low:
            raise ValueError(f"{self.gid}: high < low")


@dataclass(frozen=True)
class Encoding:
    genes: tuple[Gene, ...]

    @property
    def length(self) -> int:
        return sum(g.bits for g in self.genes)

    def decode(self, bits: str) -> tuple[float, ...]:
        if len(bits) != self.length:
            raise ValueError(f"expected {self.length} bits, got {len(bits)}")
        if any(c not in "01" for c in bits):
            raise ValueError(f"non-binary chromosome: {bits!r}")
        values = []
        i = 0
        for gene in self.genes:
            chunk = bits[i : i + gene.bits]
            i += gene.bits
            k = int(chunk, 2)
            span = (1 << gene.bits) - 1
            if span == 0:
                x = float(gene.low)
            else:
                x = gene.low + k * (gene.high - gene.low) / span
            if gene.kind == "integer":
                x = float(round(x))
            values.append(x)
        return tuple(values)

    def format_x(self, xs: tuple[float, ...]) -> str:
        parts = []
        for gene, val in zip(self.genes, xs):
            if gene.kind == "integer":
                parts.append(f"{gene.gid}={int(round(val))}")
            else:
                parts.append(f"{gene.gid}={val:.4f}")
        return ", ".join(parts)
