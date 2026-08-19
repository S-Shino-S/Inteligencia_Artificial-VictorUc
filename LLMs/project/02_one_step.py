#!/usr/bin/env python3
"""Program 2: one softmax and one attention mix (lecture numbers)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm.attention import attend  # noqa: E402
from llm.cli import build_parser, load  # noqa: E402
from llm.format import format_attention, format_softmax_demo, fmt_vec  # noqa: E402
from llm.math import softmax  # noqa: E402


def main() -> None:
    parser = build_parser("Softmax(z/T) and h = Σ α v at one position.")
    args = parser.parse_args()
    lang = load(args)

    print(lang.name)
    print()

    demo = lang.logits_demo
    if demo is None:
        print("No logits_demo in the YAML; skipping softmax.")
    else:
        print("1. From scores z to a distribution")
        print("-" * 72)
        print(format_softmax_demo(demo.labels, demo.z))
        print()

    attn = lang.attention
    if attn is None:
        raise SystemExit(f"{lang.source}: add an attention: block (tokens, alpha, values)")

    print("2. Attention: weighted sum of value vectors")
    print("-" * 72)
    print("h  =  Σ_j  α_j v_j     (α already sum to 1)")
    print()
    h = attend(attn.alpha, attn.values)
    print(format_attention(attn.tokens, attn.alpha, attn.values, h))
    print()
    parts = []
    for a, v in zip(attn.alpha, attn.values):
        parts.append(f"{a:g}·{fmt_vec(v)}")
    print("  " + "  +  ".join(parts))
    print(f"  = {fmt_vec(h)}")
    if (
        list(attn.alpha) == [0.1, 0.8, 0.1]
        and abs(h[0] - 0.2) < 1e-9
        and abs(h[1] - 1.7) < 1e-9
    ):
        print()
        print("Lecture check: 0.1·(1, 0) + 0.8·(0, 2) + 0.1·(1, 1) = (0.2, 1.7).")
        print("Almost all the weight went to gato: come looks at the subject.")

    if attn.scores is not None:
        print()
        print("3. If you start from scores instead of α, softmax comes first")
        print("-" * 72)
        a = softmax(list(attn.scores))
        print(f"scores = {fmt_vec(attn.scores, 2)}")
        print(f"α      = {fmt_vec(a, 2)}")


if __name__ == "__main__":
    main()
