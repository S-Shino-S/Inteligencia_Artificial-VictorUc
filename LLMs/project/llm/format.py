"""Pretty-print tokens, distributions, attention, and generations."""

from __future__ import annotations

from llm.counts import context_key, count_ngrams, dist_from_counts
from llm.data import Language
from llm.math import softmax


def fmt_num(v: float, width: int = 6) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
        return f"{int(round(v)):{width}d}"
    return f"{v:{width}.3f}"


def fmt_vec(xs: list[float] | tuple[float, ...], digits: int = 1) -> str:
    body = ", ".join(f"{v:.{digits}f}" for v in xs)
    return "(" + body + ")"


def format_sentence(tokens: tuple[str, ...] | list[str]) -> str:
    out: list[str] = []
    for t in tokens:
        if t in ".,!?;:" and out:
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out)


def format_language(lang: Language) -> str:
    lines = [
        lang.name,
        "",
        f"vocab ({len(lang.vocab)}):  " + "  ".join(f"{i}:{t}" for i, t in enumerate(lang.vocab)),
        f"n = {lang.n}    context = {lang.context_len} token(s)    stop = {lang.stop!r}",
        f"probe = {list(lang.probe)}    lr = {lang.lr}    epochs = {lang.epochs}    seed = {lang.seed}",
    ]
    if lang.source:
        lines.append(f"file = {lang.source}")
    lines.append("")
    lines.append("corpus")
    lines.append("-" * 72)
    for i, sent in enumerate(lang.sentences, 1):
        ids = " ".join(str(lang.token_id(t)) for t in sent)
        lines.append(f"{i}.  {format_sentence(sent)}")
        lines.append(f"    tokens {list(sent)}")
        lines.append(f"    ids    [{ids}]")
    return "\n".join(lines)


def format_dist(lang: Language, probs: list[float], title: str | None = None) -> str:
    lines = [title] if title else []
    header = f"{'token':<10}  {'P':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for tok, p in zip(lang.vocab, probs):
        bar = "#" * int(round(p * 20))
        lines.append(f"{tok:<10}  {p:8.3f}  {bar}")
    return "\n".join(lines)


def format_count_table(lang: Language) -> str:
    table = count_ngrams(lang)
    lines = [f"P(next | last {lang.context_len} token(s)) from counts"]
    lines.append("-" * 72)
    if not table:
        lines.append("(no n-gram counts)")
        return "\n".join(lines)
    for ctx in sorted(table, key=lambda c: lang.ids(c)):
        counts = table[ctx]
        probs = dist_from_counts(counts)
        ctx_s = " ".join(ctx)
        bits = "  ".join(f"{tok}={p:.2f}" for tok, p in zip(lang.vocab, probs) if p > 0)
        total = sum(counts)
        lines.append(f"{ctx_s!r:20}  n={total:<3}  {bits}")
    probe_key = context_key(lang.probe, lang.n)
    if probe_key in table:
        lines.append("")
        lines.append(format_dist(lang, dist_from_counts(table[probe_key]), f"probe {list(lang.probe)}"))
    return "\n".join(lines)


def format_attention(tokens: tuple[str, ...], alpha: tuple[float, ...] | list[float], values, h: list[float]) -> str:
    lines = [
        "h  =  Σ_j  α_j v_j",
        "",
        f"{'token':<8}  {'α':>6}  v",
        "-" * 40,
    ]
    for tok, a, v in zip(tokens, alpha, values):
        lines.append(f"{tok:<8}  {a:6.2f}  {fmt_vec(v)}")
    lines.append("")
    lines.append(f"h = {fmt_vec(h)}")
    return "\n".join(lines)


def format_softmax_demo(labels: tuple[str, ...], z: tuple[float, ...], temperatures: tuple[float, ...] = (0.5, 1.0, 2.0)) -> str:
    lines = ["P_i = exp(z_i / T) / Σ_j exp(z_j / T)", "", f"z = {fmt_vec(z, 0)}"]
    header = f"{'token':<8}" + "".join(f"  T={T:<6g}" for T in temperatures)
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    cols = [softmax(list(z), T) for T in temperatures]
    for i, lab in enumerate(labels):
        row = f"{lab:<8}"
        for col in cols:
            row += f"  {col[i]:8.3f}"
        lines.append(row)
    return "\n".join(lines)


def format_generation(tokens: list[str], title: str) -> str:
    return f"{title}:  {format_sentence(tokens)}    {list(tokens)}"


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    chars = []
    for v in values[::step]:
        idx = int(round((v - lo) / span * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        chars.append(blocks[idx])
    return "".join(chars)
