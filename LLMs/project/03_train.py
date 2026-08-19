#!/usr/bin/env python3
"""Program 3: train a softmax per context, then generate."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm.cli import build_parser, load, make_rng  # noqa: E402
from llm.counts import count_ngrams, next_dist  # noqa: E402
from llm.format import format_count_table, format_dist, format_generation, format_language, sparkline  # noqa: E402
from llm.generate import generate  # noqa: E402
from llm.train import next_dist_logits, train_logits  # noqa: E402


def main() -> None:
    parser = build_parser("Train next-token softmax (NLL) and sample a continuation.")
    parser.add_argument("--greedy", action="store_true", help="Always pick arg max P")
    args = parser.parse_args()
    lang = load(args)
    lang = replace(
        lang,
        seed=args.seed if args.seed is not None else lang.seed,
        epochs=args.epochs if args.epochs is not None else lang.epochs,
        temperature=args.temperature if args.temperature is not None else lang.temperature,
    )
    rng = make_rng(lang, None)

    print(format_language(lang))
    print()
    print(format_count_table(lang))
    print()

    print("Train:  z ← z − lr · mean(P − one_hot(y)) over the corpus")
    print(f"lr = {lang.lr}    epochs = {lang.epochs}    logits start at 0 (uniform P)")
    print()
    table, hist = train_logits(lang)
    print(f"mean NLL:  start {hist[0]:.3f}    end {hist[-1]:.3f}")
    print("NLL over epochs:  " + sparkline(hist))
    print()
    p = next_dist_logits(lang, table, lang.probe)
    print(format_dist(lang, p, f"softmax after training, context {list(lang.probe)}"))
    counts = count_ngrams(lang)
    p_count = next_dist(lang, counts, lang.probe)
    come_i = lang.vocab.index("come") if "come" in lang.vocab else None
    if come_i is not None and lang.probe == ("el", "gato"):
        print()
        print(
            f"Lecture check: counts P(come | el gato) = {p_count[come_i]:.2f}; "
            f"trained softmax → {p[come_i]:.2f}."
        )

    print()
    dist_fn = lambda ctx: next_dist_logits(lang, table, ctx, temperature=1.0)
    toks = generate(
        lang,
        dist_fn,
        greedy=args.greedy,
        temperature=1.0 if args.greedy else lang.temperature,
        rng=rng,
    )
    mode = "greedy" if args.greedy else f"sample T={lang.temperature:g}"
    print(format_generation(toks, f"generate ({mode})"))


if __name__ == "__main__":
    main()
