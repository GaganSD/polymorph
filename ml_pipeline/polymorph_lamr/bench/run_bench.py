"""CLI: the answer-survival vs compression-ratio benchmark.

Standalone and GPU-free. Mines (log, question, answer) triples from a corpus (or
uses the curated fixtures), compresses each chunk with every method across a
sweep of target drop rates, and reports survival × compression ratio.

Examples:
  # offline, deterministic, no corpus needed (fixtures):
  python -m polymorph_lamr.bench.run_bench --curated

  # mine triples from real logs and compare the always-available methods:
  python -m polymorph_lamr.bench.run_bench --corpus ../data/staged ../data/bench --max-docs 200

  # add our trained pruner and LLMLingua-2:
  python -m polymorph_lamr.bench.run_bench --corpus ../data/staged \
      --lamr-ckpt ../data/modal_out/v1/ckpt-best.pt --llmlingua

  # use an API model as the judge (extraction-query variant) instead of exact-match:
  python -m polymorph_lamr.bench.run_bench --curated --llm-judge bedrock/deepseek.v3.2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .methods import default_methods
from .survival import format_report, llm_judge_survives, run_benchmark
from .triples import AnswerTriple, build_triples_from_paths, collect_log_files, curated_triples


def _parse_rates(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _gather_triples(args) -> list[AnswerTriple]:
    if args.curated or not args.corpus:
        return curated_triples()
    files: list[Path] = []
    for c in args.corpus:
        p = Path(c)
        if p.is_dir():
            files.extend(collect_log_files(p))
        elif p.is_file():
            files.append(p)
    return build_triples_from_paths(
        files,
        window_lines=args.window_lines,
        max_per_file=args.max_per_file,
        max_total=args.max_docs,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Answer-survival vs compression-ratio benchmark.")
    p.add_argument("--corpus", nargs="+", default=None, help="dirs/files of log text to mine triples from")
    p.add_argument("--curated", action="store_true", help="use the built-in fixture triples (no corpus)")
    p.add_argument("--drop-rates", type=str, default="0.1,0.3,0.5,0.7", help="comma-separated target drop rates")
    p.add_argument("--max-docs", type=int, default=200, help="cap on total triples")
    p.add_argument("--max-per-file", type=int, default=5)
    p.add_argument("--window-lines", type=int, default=40)
    p.add_argument("--lamr-ckpt", type=Path, default=None, help="LaMR checkpoint to include as a method")
    p.add_argument("--llmlingua", action="store_true", help="include LLMLingua-2 (heavy; downloads a model)")
    p.add_argument("--llm-judge", type=str, default=None,
                   help="litellm model to judge survival via the extraction query (default: offline exact-match)")
    p.add_argument("--out", type=Path, default=None, help="write the full results table as JSON here")
    args = p.parse_args(argv)

    rates = _parse_rates(args.drop_rates)
    triples = _gather_triples(args)
    if not triples:
        print("no triples found (empty corpus / no extractable needles); try --curated")
        return 1

    methods = default_methods(lamr_ckpt=args.lamr_ckpt, include_llmlingua=args.llmlingua)

    survival_fn = None
    if args.llm_judge:
        model = args.llm_judge
        survival_fn = lambda t, comp: llm_judge_survives(t.question, comp, t.answer, model)  # noqa: E731

    if survival_fn is None:
        results, skipped = run_benchmark(triples, methods, rates)
    else:
        results, skipped = run_benchmark(triples, methods, rates, survival_fn)

    report = format_report(results, skipped, triples, rates)
    print(report)

    if args.out:
        payload = {
            "drop_rates": rates,
            "n_triples": len(triples),
            "judge": args.llm_judge or "exact-match",
            "results": {name: [asdict(r) for r in rows] for name, rows in results.items()},
            "skipped": skipped,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
