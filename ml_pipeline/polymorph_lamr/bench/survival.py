"""Answer-survival metric + the rate-distortion sweep and report.

The benchmark maps **answer survival** (did the needle fact survive compression?)
against **compression ratio** (tokens in / tokens out) for each method. The
primary survival test is a GPU-free, deterministic exact-match: after collapsing
whitespace and case, does the answer substring still appear in the compressed
text? An optional LLM judge (``--llm-judge``, via litellm) instead asks a model
the extraction question over the compressed payload and checks the answer — the
"feed a model + extraction query" variant — but the default is offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .methods import CompressionMethod, token_count
from .triples import AnswerTriple

# A survival test: (triple, compressed_text) -> survived?  The default is the
# offline exact-match; --llm-judge swaps in a model-based one.
SurvivalFn = Callable[[AnswerTriple, str], bool]

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def answer_survives(answer: str, compressed: str) -> bool:
    """Exact-match survival: whitespace-collapsed, case-insensitive substring."""
    return _norm(answer) in _norm(compressed)


def compression_ratio(orig_text: str, comp_text: str) -> float:
    """tokens_in / tokens_out (>= 1 means smaller; higher = more compression)."""
    out = token_count(comp_text)
    return token_count(orig_text) / out if out else float("inf")


def achieved_drop_rate(orig_text: str, comp_text: str) -> float:
    o = token_count(orig_text)
    return 1.0 - (token_count(comp_text) / o) if o else 0.0


@dataclass
class MethodRow:
    method: str
    target_drop_rate: float
    survival: float            # fraction of triples whose answer survived
    mean_ratio: float          # mean tokens_in / tokens_out
    mean_achieved_drop: float  # mean realized drop fraction
    n: int


def _default_survival(t: AnswerTriple, comp: str) -> bool:
    return answer_survives(t.answer, comp)


def evaluate_method(
    method: CompressionMethod,
    triples: list[AnswerTriple],
    drop_rates: list[float],
    survival_fn: SurvivalFn = _default_survival,
) -> list[MethodRow]:
    """Sweep ``drop_rates`` for one method. Non-tunable methods are evaluated once
    (at their natural operating point) and reported as a single row."""
    rates = drop_rates if method.tunable else [0.0]  # 0.0 = "ignored" sentinel
    rows: list[MethodRow] = []
    for r in rates:
        survived = 0
        ratios: list[float] = []
        drops: list[float] = []
        for t in triples:
            comp = method.compress(t.text, r)
            if survival_fn(t, comp):
                survived += 1
            ratios.append(compression_ratio(t.text, comp))
            drops.append(achieved_drop_rate(t.text, comp))
        n = len(triples)
        rows.append(
            MethodRow(
                method=method.name,
                target_drop_rate=r,
                survival=survived / n if n else 0.0,
                mean_ratio=sum(ratios) / len(ratios) if ratios else 0.0,
                mean_achieved_drop=sum(drops) / len(drops) if drops else 0.0,
                n=n,
            )
        )
    return rows


def survival_vector(
    method: CompressionMethod,
    triples: list[AnswerTriple],
    drop_rate: float,
    survival_fn: SurvivalFn = _default_survival,
) -> list[bool]:
    """Per-triple survival booleans for one method at one drop rate (for paired
    significance testing — McNemar needs the aligned per-item outcomes)."""
    return [survival_fn(t, method.compress(t.text, drop_rate)) for t in triples]


def mcnemar(a: list[bool], b: list[bool]) -> dict[str, float]:
    """Paired McNemar test comparing method A vs B on the SAME triples.

    Returns the discordant counts and a two-sided p-value. ``b01`` = A wrong / B
    right, ``b10`` = A right / B wrong. We use the exact binomial test on the
    discordant pairs (robust for the small counts a 400-triple survival delta
    produces; the chi-square approximation is unreliable when b01+b10 < ~25).
    A significant result with b10 > b01 means A preserves more needles than B.
    """
    from math import comb

    if len(a) != len(b):
        raise ValueError("survival vectors must be aligned (same triples/order)")
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        # Two-sided exact binomial p at theta=0.5.
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
        p = min(1.0, 2.0 * tail)
    return {"b01_a_worse": float(b01), "b10_a_better": float(b10), "n_discordant": float(n), "p_value": p}


def run_benchmark(
    triples: list[AnswerTriple],
    methods: list[CompressionMethod],
    drop_rates: list[float],
    survival_fn: SurvivalFn = _default_survival,
) -> tuple[dict[str, list[MethodRow]], dict[str, str]]:
    """Returns (results_by_method, skipped_by_method_with_reason)."""
    results: dict[str, list[MethodRow]] = {}
    skipped: dict[str, str] = {}
    for m in methods:
        ok, reason = m.available()
        if not ok:
            skipped[m.name] = reason
            continue
        results[m.name] = evaluate_method(m, triples, drop_rates, survival_fn)
    return results, skipped


def format_report(
    results: dict[str, list[MethodRow]],
    skipped: dict[str, str],
    triples: list[AnswerTriple],
    drop_rates: list[float],
) -> str:
    lines: list[str] = []
    lines.append("== Polymorph answer-survival benchmark ==")
    lines.append(f"triples: {len(triples)}   target drop rates: {drop_rates}")
    # Fact-type breakdown.
    by_type: dict[str, int] = {}
    for t in triples:
        by_type[t.fact_type] = by_type.get(t.fact_type, 0) + 1
    lines.append("fact types: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    lines.append("")
    lines.append("survival % (compression ratio x) at each target drop rate:")
    header = "  method".ljust(20) + "".join(f"R={r:<10.2f}" for r in drop_rates)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, rows in results.items():
        if not rows:
            continue
        if len(rows) == 1 and rows[0].target_drop_rate == 0.0 and len(drop_rates) != 1:
            # non-tunable: one operating point, printed once with its achieved drop.
            row = rows[0]
            cell = f"{row.survival * 100:4.0f}% ({row.mean_ratio:.2f}x @drop{row.mean_achieved_drop:.2f})"
            lines.append("  " + name.ljust(18) + cell + "   [not rate-tunable]")
            continue
        cells = ""
        by_r = {row.target_drop_rate: row for row in rows}
        for r in drop_rates:
            row = by_r.get(r)
            if row is None:
                cells += "—".ljust(12)
            else:
                cells += f"{row.survival * 100:3.0f}%({row.mean_ratio:.1f}x)".ljust(12)
        lines.append("  " + name.ljust(18) + cells)
    if skipped:
        lines.append("")
        lines.append("skipped methods:")
        for name, reason in skipped.items():
            lines.append(f"  {name}: {reason}")
    lines.append("")
    lines.append(
        "Read: higher survival at higher compression (more x / higher R) is better.\n"
        "  'random' is the floor; a real ranker must keep needles the floor drops.\n"
        "  survival is exact-match of the answer needle in the compressed text."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM judge (feed a model the compressed payload + the question)
# ---------------------------------------------------------------------------

def llm_judge_survives(
    question: str, compressed: str, gold_answer: str, model: str
) -> bool:  # pragma: no cover - network/env dependent
    """Judge whether the gold fact SURVIVED compression, robust to paraphrase.

    Earlier design ("answer the open question, then substring-match the gold")
    was broken for log facts: the question is generic ("what event was reported?")
    so the model enumerates many events, the gold phrase is paraphrased
    ("Registered signal handlers for [TERM,HUP,INT]" vs "Signal handlers
    registered for TERM, HUP and INT"), and max_tokens truncates the reply — so a
    fact that plainly survived scored 0. Instead we ask a direct YES/NO entailment
    on the gold fact against the compressed excerpt: does the excerpt state or let
    you directly answer this specific fact? This tests survival of the FACT (the
    thing we care about), not surface-form reproduction, and needs only one token.
    """
    import litellm

    prompt = (
        "You are checking whether a specific fact survived log compression. Below "
        "is a compressed log excerpt, then a FACT. Answer YES if the fact is stated "
        "in the excerpt or can be directly read from it (allowing paraphrase / "
        "different wording / formatting). Answer NO if the information is absent. "
        "Reply with ONLY 'YES' or 'NO'.\n\n"
        f"--- COMPRESSED LOG ---\n{compressed}\n--- END ---\n\n"
        f"FACT: {gold_answer}\n\nIs this fact present or directly answerable? Answer:"
    )
    resp = litellm.completion(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=4
    )
    answer = _norm(resp["choices"][0]["message"]["content"])
    return answer.startswith("yes")
