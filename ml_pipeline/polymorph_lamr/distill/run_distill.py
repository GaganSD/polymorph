"""CLI: distill a directory tree of logs/traces into JSONL training pairs.

Primary path is the open-weight multi-teacher ENSEMBLE (E3): each chunk is
compressed by several teachers and the best-QC output is kept as the training
target. Default teachers (see `providers.py`):

* deepseek-v32 via AWS Bedrock  -> AWS credential chain + AWS_REGION
* kimi-k2.6:free via OpenRouter  -> needs OPENROUTER_API_KEY

    AWS_REGION=us-east-1 OPENROUTER_API_KEY=sk-or-... \
        python -m polymorph_lamr.distill.run_distill \
        --in data/raw --out data/distilled.jsonl --concurrency 8

A teacher missing its key (or rate-limited) is dropped from per-chunk selection
rather than failing the run. A legacy two-teacher (Claude + GPT-4o) mode is kept
for back-compat: --mode pair.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .chunker import chunk, detect_mode
from .client import (
    DistillConfig,
    EnsembleConfig,
    TeacherSpec,
    default_teachers,
    distill_ensemble_many,
    distill_many,
    write_jsonl,
)
from .prompts import LOG_TRACE_EXTRACTIVE

# Logs/traces first; keep code/prose exts so the tool is still general.
_DEFAULT_EXTS = {".log", ".txt", ".json", ".jsonl", ".py", ".md", ".rs", ".ts", ".tsx", ".js"}


def _iter_files(root: Path, exts: set[str]) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts)
    # Skip log-parser artifacts that ship alongside TrainTicket logs.
    return [
        f
        for f in files
        if not f.name.endswith(("_structured.csv", "_templates.csv"))
        and not f.name.startswith("potentialAnomalies")
    ]


def _load_items(files: list[Path], max_tokens: int) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        mode = detect_mode(str(f), text)
        for idx, c in enumerate(chunk(text, max_tokens=max_tokens, mode=mode)):
            items.append((c, str(f), idx))
    return items


def _parse_teachers(specs: list[str] | None) -> list[TeacherSpec]:
    """Parse `--teachers name=spec` entries into routed TeacherSpecs.

    The spec is provider-prefixed and routed via `TeacherSpec.from_spec`, e.g.
    `--teachers qwen3=vercel/alibaba/qwen3.7-max kimi=openrouter/moonshotai/kimi-k2.6:free`.
    A bare spec (no `=`) takes its last path segment as the label.
    """
    if not specs:
        return default_teachers()
    out: list[TeacherSpec] = []
    for s in specs:
        if "=" in s:
            name, model = s.split("=", 1)
        else:
            name, model = s.split("/")[-1], s
        out.append(TeacherSpec.from_spec(name.strip(), model.strip()))
    return out


async def _run(args: argparse.Namespace) -> int:
    files = _iter_files(Path(args.input), set(args.exts))
    items = _load_items(files, max_tokens=args.max_tokens)
    if not items:
        print("no items found", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        # Cap chunks for smoke runs / rate-limited free tiers (e.g. OpenRouter
        # free models). Deterministic prefix — items are file/chunk-ordered.
        items = items[: args.limit]

    results: list = []
    total_cost = 0.0

    if args.mode == "ensemble":
        teachers = _parse_teachers(args.teachers)
        missing = sorted(
            {
                t.api_key_env
                for t in teachers
                if t.api_key_env and not os.environ.get(t.api_key_env)
            }
        )
        if missing:
            print(
                f"[warn] missing API key(s) {', '.join(missing)}; the teacher(s) "
                "needing them will error and be dropped from best-QC selection. "
                "Export them before a real run.",
                file=sys.stderr,
            )
        cfg = EnsembleConfig(
            teachers=teachers,
            prompt_template=LOG_TRACE_EXTRACTIVE,
            num_retries=args.retries,
            request_timeout_s=args.timeout,
            max_tokens=args.output_max_tokens,
            temperature=args.temperature,
            failure_policy=args.failure_policy,
        )
        print(
            f"distilling {len(items)} chunks from {len(files)} files via "
            f"{len(teachers)} teachers "
            f"({', '.join(t.name for t in teachers)}); concurrency={args.concurrency}"
        )
        async for r in distill_ensemble_many(items, cfg=cfg, concurrency=args.concurrency):
            results.append(r)
            total_cost += r.cost_usd
            if r.errors:
                print(f"[warn] {r.src_path}#{r.chunk_id}: {r.errors}", file=sys.stderr)
            if len(results) % 25 == 0:
                print(f"  progress: {len(results)}/{len(items)} cost=${total_cost:.4f}")
    else:  # legacy pair
        cfg = DistillConfig(
            claude_model=args.claude_model,
            gpt_model=args.gpt_model,
            num_retries=args.retries,
            request_timeout_s=args.timeout,
            max_tokens=args.output_max_tokens,
            temperature=args.temperature,
            failure_policy=args.failure_policy,
        )
        print(f"distilling {len(items)} chunks (pair mode) from {len(files)} files")
        async for r in distill_many(items, cfg=cfg, concurrency=args.concurrency):
            results.append(r)
            total_cost += r.cost_usd
            if r.errors:
                print(f"[warn] {r.src_path}#{r.chunk_id} errors: {r.errors}", file=sys.stderr)

    write_jsonl(results, Path(args.output))
    print(f"wrote {len(results)} records to {args.output} (cost ${total_cost:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distill logs/traces into compressed training pairs.")
    p.add_argument("--in", dest="input", required=True, help="source file or directory")
    p.add_argument("--out", dest="output", required=True, help="output JSONL path")
    p.add_argument("--mode", choices=["ensemble", "pair"], default="ensemble",
                   help="ensemble = open-weight teachers (default); pair = legacy Claude+GPT-4o")
    p.add_argument("--teachers", nargs="*", default=None,
                   help="ensemble teachers as name=spec (default: deepseek-v32 Bedrock + kimi OpenRouter)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number of chunks (0 = all); use for smoke runs and rate-limited free tiers")
    p.add_argument("--max-tokens", type=int, default=512, help="per-chunk cap")
    p.add_argument("--exts", nargs="*", default=sorted(_DEFAULT_EXTS))
    p.add_argument("--claude-model", default="anthropic/claude-3-5-sonnet-latest", help="pair mode only")
    p.add_argument("--gpt-model", default="openai/gpt-4o", help="pair mode only")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--output-max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--failure-policy", choices=["record", "raise"], default="record")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
