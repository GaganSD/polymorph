"""Semantic measuring stick: LLM-judge vs exact-match answer survival.

The exact-match survival test (``survival.answer_survives``) is strict: the gold
needle must appear verbatim (whitespace/case-normalized) in the compressed text.
That undercounts a compressor that preserves the *fact* while dropping the exact
surface form — e.g. it keeps "returned a 500" but the gold answer was "500", or
it paraphrases a root-cause phrase. The LLM judge measures the thing we actually
care about: can a downstream model still ANSWER the extraction question from the
compressed payload? (``survival.llm_judge_survives`` — reused here, not
reimplemented.)

This CLI runs a chosen set of methods at one (or more) drop rates over a
deterministically-subsampled slice of the held-out triples, scores survival with
BOTH exact-match and the judge, and prints a per-method comparison. The headline
number is the **judge-recovered** count: cases the judge marks SURVIVED that
exact-match marked dead — the semantic recovery exact-match is blind to.

Cost is bounded: ``--sample`` (default 30) subsamples triples deterministically
and ``--drop-rates`` defaults to a single rate 0.5. Total judge calls =
n_methods * n_rates * sample.

Judge wiring (Vercel AI Gateway, OpenAI-compatible, litellm):
  export OPENAI_API_KEY=$VERCEL_AI_GATEWAY_KEY
  export OPENAI_BASE_URL=https://ai-gateway.vercel.sh/v1
  --judge-model openai/alibaba/qwen3.7-max   (the default)
This script auto-bootstraps that wiring from VERCEL_AI_GATEWAY_KEY if OPENAI_*
are unset, so it works out of the box with the repo .env loaded. Any litellm
model string the environment can authenticate also works via --judge-model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .build_heldout import load_triples
from .methods import (
    CompressionMethod,
    DeterministicDedup,
    KeepSeverityHeuristic,
    LaMRMethod,
    LLMLingua2Method,
    RandomDropFloor,
    token_count,
)
from .survival import answer_survives, llm_judge_survives
from .triples import AnswerTriple

DEFAULT_TRIPLES = Path("../data/bench/heldout_triples.json")
DEFAULT_OUT = Path("../data/bench/judge_result.json")
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"  # needs ANTHROPIC_API_KEY in env
VERCEL_BASE_URL = "https://ai-gateway.vercel.sh/v1"


def _bootstrap_judge_env() -> str:
    """Wire litellm's OpenAI-compatible client to the Vercel AI Gateway from the
    repo .env if the OPENAI_* vars aren't already set. Returns a short status
    string describing what was used (never the key itself)."""
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return f"OPENAI_BASE_URL={os.environ['OPENAI_BASE_URL']} (pre-set)"
    vkey = os.environ.get("VERCEL_AI_GATEWAY_KEY")
    if vkey:
        os.environ.setdefault("OPENAI_API_KEY", vkey)
        os.environ.setdefault("OPENAI_BASE_URL", VERCEL_BASE_URL)
        return f"OPENAI_BASE_URL={VERCEL_BASE_URL} (from VERCEL_AI_GATEWAY_KEY)"
    return "no OPENAI_* / VERCEL_AI_GATEWAY_KEY in env (judge will likely fail)"


def _subsample(triples: list[AnswerTriple], n: int) -> list[AnswerTriple]:
    """Deterministic subsample of ``n`` triples. Sorts by a stable content hash so
    the same N triples are always picked (reproducible cost + comparable runs),
    independent of file order."""
    if n <= 0 or n >= len(triples):
        return list(triples)
    keyed = sorted(
        triples,
        key=lambda t: hashlib.sha1(f"{t.doc_id}|{t.fact_type}|{t.answer}".encode()).hexdigest(),
    )
    return keyed[:n]


def _build_methods(names: list[str], lamr_ckpt: Path | None) -> list[CompressionMethod]:
    catalog: dict[str, CompressionMethod] = {
        "deterministic": DeterministicDedup(),
        "keep-severity": KeepSeverityHeuristic(),
        "random": RandomDropFloor(),
        "random+floor": RandomDropFloor(floor=True),
        "llmlingua2": LLMLingua2Method(),
    }
    if lamr_ckpt is not None:
        catalog["lamr"] = LaMRMethod(ckpt=Path(lamr_ckpt))
        catalog["lamr+floor"] = LaMRMethod(ckpt=Path(lamr_ckpt), floor=True)
        # Span-aware (Word+Max) variants — the fragmentation-fixed decode the Rust
        # runtime uses; this is the configuration the SOTA gate is actually run in.
        catalog["lamr+span"] = LaMRMethod(ckpt=Path(lamr_ckpt), span="word")
        catalog["lamr+span+floor"] = LaMRMethod(ckpt=Path(lamr_ckpt), span="word", floor=True)
    out: list[CompressionMethod] = []
    for n in names:
        if n not in catalog:
            raise SystemExit(
                f"unknown method '{n}'. choices: {sorted(catalog)}"
                + ("" if lamr_ckpt else " (pass --lamr-ckpt to enable lamr/lamr+floor)")
            )
        out.append(catalog[n])
    return out


def _achieved_ratio(original: str, compressed: str) -> float:
    """cl100k-token compression ratio orig/comp, measured on the SAME yardstick for
    every method (so a ModernBERT-tokenized LaMR and a line-level keep-severity are
    comparable). Guards a zero-token compression."""
    return token_count(original) / max(1, token_count(compressed))


def _compress_to_ratio(
    method: CompressionMethod, text: str, target_ratio: float, iters: int = 14
) -> tuple[str, float]:
    """Bisect a tunable method's target drop rate until its achieved cl100k-token
    ratio is ~``target_ratio``. Returns (compressed_text, achieved_ratio) for the
    closest rate found. Non-tunable methods (e.g. dedup) ignore the rate — we just
    compress once and report whatever ratio they land on.

    Monotonic assumption: higher drop rate -> fewer surviving tokens -> higher
    ratio. True for every rate-tunable method here (drop-order / keep-budget).
    """
    if not getattr(method, "tunable", True):
        comp = method.compress(text, 0.0)
        return comp, _achieved_ratio(text, comp)
    lo, hi = 0.0, 0.99
    best: tuple[str, float] | None = None
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        comp = method.compress(text, mid)
        r = _achieved_ratio(text, comp)
        if best is None or abs(r - target_ratio) < abs(best[1] - target_ratio):
            best = (comp, r)
        if r < target_ratio:   # not compressed enough -> drop more
            lo = mid
        else:                  # over-compressed -> drop less
            hi = mid
    assert best is not None
    return best


@dataclass
class JudgeCell:
    method: str
    drop_rate: float       # target drop rate, or -1.0 in iso-ratio mode
    n: int
    mean_ratio: float      # mean achieved cl100k-token compression ratio (orig/comp)
    exact_survived: int
    judge_survived: int
    judge_recovered: int   # judge SURVIVED & exact NO  (semantic recovery)
    judge_lost: int        # exact YES & judge NO  (surface kept, fact not answerable)
    judge_errors: int      # judge call raised / returned no decision


def _evaluate(
    method: CompressionMethod,
    triples: list[AnswerTriple],
    drop_rate: float,
    judge_model: str,
    iso_ratio: float | None = None,
) -> tuple[JudgeCell, list[dict]]:
    """Evaluate one method over ``triples``. In matched-rate mode every item is
    compressed at ``drop_rate``; in iso-ratio mode (``iso_ratio`` set) each item's
    drop rate is bisected so its achieved compression ratio is ~``iso_ratio``, so
    survival is compared at EQUAL compression rather than equal target rate."""
    exact_n = judge_n = recovered = lost = errors = 0
    ratio_sum = 0.0
    per_item: list[dict] = []
    for t in triples:
        if iso_ratio is not None:
            comp, ratio = _compress_to_ratio(method, t.text, iso_ratio)
        else:
            comp = method.compress(t.text, drop_rate)
            ratio = _achieved_ratio(t.text, comp)
        ratio_sum += ratio
        exact = answer_survives(t.answer, comp)
        try:
            judge = llm_judge_survives(t.question, comp, t.answer, judge_model)
            judge_err = False
        except Exception as e:  # noqa: BLE001 - keep the sweep going; record the failure
            judge = False
            judge_err = True
            errors += 1
            print(f"    [judge error] {method.name} {t.doc_id}: {type(e).__name__}: {e}", file=sys.stderr)
        exact_n += int(exact)
        judge_n += int(judge)
        if judge and not exact:
            recovered += 1
        if exact and not judge and not judge_err:
            lost += 1
        per_item.append(
            {
                "doc_id": t.doc_id,
                "fact_type": t.fact_type,
                "answer": t.answer,
                "ratio": round(ratio, 3),
                "exact": exact,
                "judge": judge,
                "judge_error": judge_err,
            }
        )
    cell = JudgeCell(
        method=method.name,
        drop_rate=(-1.0 if iso_ratio is not None else drop_rate),
        n=len(triples),
        mean_ratio=(ratio_sum / len(triples)) if triples else 0.0,
        exact_survived=exact_n,
        judge_survived=judge_n,
        judge_recovered=recovered,
        judge_lost=lost,
        judge_errors=errors,
    )
    return cell, per_item


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den) if den else 0.0:4.0f}%"


def _format_table(cells: list[JudgeCell]) -> str:
    lines: list[str] = []
    lines.append("== Semantic survival: exact-match vs LLM judge ==")
    hdr = (
        "  " + "method".ljust(18) + "rate".ljust(6) + "ratio".ljust(7) + "n".ljust(5)
        + "exact".ljust(8) + "judge".ljust(8)
        + "recovered".ljust(11) + "lost".ljust(7) + "err"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for c in cells:
        rate_s = "iso" if c.drop_rate < 0 else f"{c.drop_rate:.2f}"
        lines.append(
            "  "
            + c.method.ljust(18)
            + rate_s.ljust(6)
            + f"{c.mean_ratio:.2f}x".ljust(7)
            + str(c.n).ljust(5)
            + _pct(c.exact_survived, c.n).ljust(8)
            + _pct(c.judge_survived, c.n).ljust(8)
            + f"{c.judge_recovered} ({_pct(c.judge_recovered, c.n).strip()})".ljust(11)
            + str(c.judge_lost).ljust(7)
            + str(c.judge_errors)
        )
    lines.append("")
    lines.append(
        "Read: 'recovered' = judge says the fact is answerable but exact-match\n"
        "  missed the verbatim needle — the semantic survival exact-match is blind to.\n"
        "  'lost' = needle present verbatim but the judge could not answer (surface\n"
        "  survived, fact did not). 'err' = judge call failed (excluded from judge%)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lamr-judge-bench",
        description="Compare exact-match vs LLM-judge answer survival on held-out triples.",
    )
    p.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES, help="held-out triples JSON")
    p.add_argument("--sample", type=int, default=30, help="deterministic subsample size (0 = all)")
    p.add_argument("--drop-rates", type=str, default="0.5", help="comma-separated target drop rates")
    p.add_argument(
        "--iso-ratio",
        type=str,
        default=None,
        help="comma-separated target compression ratios (e.g. '3,5'). When set, each "
        "item's drop rate is bisected to hit the ratio, so survival is compared at "
        "EQUAL compression instead of equal target rate (overrides --drop-rates).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="keep-severity,random,deterministic",
        help="comma-separated method names (deterministic,keep-severity,random,random+floor,llmlingua2,lamr,lamr+floor)",
    )
    p.add_argument("--lamr-ckpt", type=Path, default=None, help="enable lamr/lamr+floor methods")
    p.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="litellm model string for the judge")
    p.add_argument("--semantic-only", action="store_true", help="restrict to semantic: needles (the discriminating class)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="write results JSON here")
    p.add_argument(
        "--stats",
        action="store_true",
        help="after the run, compute per-domain/per-fact_type survival, McNemar "
        "paired tests (lamr+span vs keep-severity / llmlingua2), and bootstrap "
        "95%% CIs from the per-item results, and print the defensible-eval table.",
    )
    p.add_argument("--stats-metric", choices=["judge", "exact"], default="judge", help="metric for the --stats table")
    p.add_argument("--stats-seed", type=int, default=1234, help="bootstrap seed (determinism)")
    args = p.parse_args(argv)

    judge_status = _bootstrap_judge_env()
    rates = [float(x) for x in args.drop_rates.split(",") if x.strip()]
    iso_targets = (
        [float(x) for x in args.iso_ratio.split(",") if x.strip()]
        if args.iso_ratio
        else None
    )
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]

    triples = load_triples(args.triples)
    if args.semantic_only:
        triples = [t for t in triples if t.fact_type.startswith("semantic:")]
    sample = _subsample(triples, args.sample)

    methods = _build_methods(method_names, args.lamr_ckpt)
    # Drop methods that aren't available (e.g. llmlingua2 not installed), with a reason.
    live: list[CompressionMethod] = []
    skipped: dict[str, str] = {}
    for m in methods:
        ok, reason = m.available()
        if ok:
            live.append(m)
        else:
            skipped[m.name] = reason

    sweep = iso_targets if iso_targets is not None else rates
    mode = "iso-ratio" if iso_targets is not None else "matched-rate"
    print(f"judge: model={args.judge_model}  {judge_status}")
    print(f"triples: loaded={len(triples)}  sampled={len(sample)}  mode={mode}  sweep={sweep}")
    print(f"methods: {[m.name for m in live]}" + (f"  skipped={skipped}" if skipped else ""))
    print(f"projected judge calls: {len(live) * len(sweep) * len(sample)}\n")

    cells: list[JudgeCell] = []
    per_item_all: dict[str, list[dict]] = {}
    for val in sweep:
        for m in live:
            if iso_targets is not None:
                print(f"  evaluating {m.name} @ iso-ratio={val:.2f}x over {len(sample)} triples ...", flush=True)
                cell, items = _evaluate(m, sample, 0.0, args.judge_model, iso_ratio=val)
                per_item_all[f"{m.name}@iso{val}"] = items
            else:
                print(f"  evaluating {m.name} @ drop={val:.2f} over {len(sample)} triples ...", flush=True)
                cell, items = _evaluate(m, sample, val, args.judge_model)
                per_item_all[f"{m.name}@{val}"] = items
            cells.append(cell)

    print("\n" + _format_table(cells))

    total_recovered = sum(c.judge_recovered for c in cells)
    print(f"\nTOTAL judge-recovered-but-exact-missed across all cells: {total_recovered}")

    payload = {
        "judge_model": args.judge_model,
        "judge_wiring": judge_status,
        "triples_file": str(args.triples),
        "n_loaded": len(triples),
        "n_sampled": len(sample),
        "semantic_only": args.semantic_only,
        "mode": mode,
        "drop_rates": rates,
        "iso_ratios": iso_targets,
        "methods": [m.name for m in live],
        "skipped": skipped,
        "cells": [c.__dict__ for c in cells],
        "total_judge_recovered": total_recovered,
        "per_item": per_item_all,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    if args.stats:
        from .stats import analyze, format_stats

        analysis = analyze(per_item_all, seed=args.stats_seed)
        print("\n" + format_stats(analysis, metric=args.stats_metric))
        stats_out = args.out.with_name(args.out.stem + "_stats.json")
        stats_out.write_text(json.dumps(analysis, indent=2))
        print(f"\nwrote {stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
