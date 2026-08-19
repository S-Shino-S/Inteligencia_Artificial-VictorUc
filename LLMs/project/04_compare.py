#!/usr/bin/env python3
"""Program 4: temperature, greedy vs sample, window, unseen context."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm.attention import attend, causal_self_attention  # noqa: E402
from llm.cli import ROOT as PROJ  # noqa: E402
from llm.counts import context_key, count_ngrams, next_dist  # noqa: E402
from llm.data import load_language  # noqa: E402
from llm.format import format_dist, format_generation, format_softmax_demo, fmt_vec  # noqa: E402
from llm.generate import generate  # noqa: E402

DATA = PROJ / "data"


def attention_demo() -> None:
    lang = load_language(DATA / "gato.yaml")
    attn = lang.attention
    assert attn is not None
    print("1. Lecture attention")
    print("-" * 72)
    h = attend(attn.alpha, attn.values)
    print(f"α = {fmt_vec(attn.alpha, 1)}    h = {fmt_vec(h)}")
    print("Given α, not learned: this is the mix the slide computes by hand.")
    print()
    print("Same three vectors, but α from causal dots x_t · x_j (not the lecture row):")
    _outs, alphas = causal_self_attention(attn.values)
    for tok, a in zip(attn.tokens, alphas):
        print(f"  {tok:<6}  α = {fmt_vec(a, 2)}")
    print("The recipe is the same (softmax, then Σ α v). The scores are different, so α is too.")


def temperature_demo() -> None:
    lang = load_language(DATA / "gato.yaml")
    demo = lang.logits_demo
    assert demo is not None
    print("2. Temperature on z = (2, 0, 0)")
    print("-" * 72)
    print(format_softmax_demo(demo.labels, demo.z))
    print()
    print("T = 0.5 piles mass on come. T = 2 flattens toward . and el.")


def sample_demo() -> None:
    lang = load_language(DATA / "gato.yaml")
    table = count_ngrams(lang)
    dist_fn = lambda ctx: next_dist(lang, table, ctx)
    print("3. Greedy vs sampling (count model)")
    print("-" * 72)
    print("Prefix: el gato. Stop token is '.'")
    print()
    greedy = generate(lang, dist_fn, greedy=True, rng=random.Random(0))
    print(format_generation(greedy, "greedy"))
    print()
    for T, seed in ((1.0, 0), (1.0, 1), (2.0, 0)):
        toks = generate(lang, dist_fn, greedy=False, temperature=T, rng=random.Random(seed))
        print(format_generation(toks, f"sample T={T:g} seed={seed}"))
    print()
    print("Greedy always takes come (0.75). Sampling can draw '.' (0.25) and stop early.")


def window_demo() -> None:
    lang = load_language(DATA / "gato.yaml")
    table = count_ngrams(lang)
    print("4. The window is the last n−1 tokens")
    print("-" * 72)
    long_ctx = ["el", "gato", "come"]
    used = context_key(long_ctx, lang.n)
    print(f"n = {lang.n}    written = {long_ctx}    model sees {list(used)}")
    print()
    print(format_dist(lang, next_dist(lang, table, long_ctx), f"P(next | {list(used)})"))
    print()
    print("el at the left has already fallen out. An LLM with a finite L does the same.")


def unseen_demo() -> None:
    gato = load_language(DATA / "gato.yaml")
    more = load_language(DATA / "more.yaml")
    print("5. Unseen context")
    print("-" * 72)
    table = count_ngrams(gato)
    ctx = ("el", "perro")
    print("gato.yaml never saw 'perro'. P(next | el perro) is uniform:")
    print(format_dist(gato, next_dist(gato, table, ctx), None))
    print()
    table2 = count_ngrams(more)
    print("more.yaml includes el perro come . — now the mass sits on come:")
    print(format_dist(more, next_dist(more, table2, ctx), None))
    print()
    print("Fluent uniform noise is the tiny analogue of an LLM guessing when the context is new.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare temperature, sampling, window, unseen context.")
    parser.add_argument(
        "--only",
        choices=("attention", "temperature", "sample", "window", "unseen", "all"),
        default="all",
    )
    args = parser.parse_args()
    demos = {
        "attention": attention_demo,
        "temperature": temperature_demo,
        "sample": sample_demo,
        "window": window_demo,
        "unseen": unseen_demo,
    }
    names = list(demos) if args.only == "all" else [args.only]
    for i, name in enumerate(names):
        if i:
            print()
            print()
        demos[name]()


if __name__ == "__main__":
    main()
