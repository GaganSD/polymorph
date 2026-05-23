"""CLI: distill a directory tree of source files into JSONL pairs.

Usage:
    python -m polymorph_lamr.distill.run_distill \
        --in data/raw --out data/distilled.jsonl --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .chunker import chunk, detect_mode
from .client import DistillConfig, distill_many, write_jsonl


_DEFAULT_EXTS = {".py", ".md", ".rs", ".ts", ".tsx", ".js", ".json", ".txt"}


def _iter_files(root: Path, exts: set[str]) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts)


def _load_items(files: list[Path], max_tokens: int) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        mode = detect_mode(str(f), text)
        for idx, c in enumerate(chunk(text, max_tokens=max_tokens, mode=mode)):
            items.append((c, str(f), idx))
    return items


async def _run(args: argparse.Namespace) -> int:
    files = _iter_files(Path(args.input), set(args.exts))
    items = _load_items(files, max_tokens=args.max_tokens)
    if not items:
        print("no items found", file=sys.stderr)
        return 1

    cfg = DistillConfig(
        claude_model=args.claude_model,
        gpt_model=args.gpt_model,
        num_retries=args.retries,
        request_timeout_s=args.timeout,
    )

    results = []
    total_cost = 0.0
    print(f"distilling {len(items)} chunks from {len(files)} files (concurrency={args.concurrency})")
    async for r in distill_many(items, cfg=cfg, concurrency=args.concurrency):
        results.append(r)
        total_cost += r.cost_usd
        if r.errors:
            print(f"[warn] {r.src_path}#{r.chunk_id} errors: {r.errors}", file=sys.stderr)
        if len(results) % 25 == 0:
            print(f"  progress: {len(results)}/{len(items)} cost=${total_cost:.4f}")

    write_jsonl(results, Path(args.output))
    print(f"wrote {len(results)} records to {args.output} (cost ${total_cost:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distill files into Claude/GPT-4o compressed pairs.")
    p.add_argument("--in", dest="input", required=True, help="source file or directory")
    p.add_argument("--out", dest="output", required=True, help="output JSONL path")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512, help="per-chunk cap")
    p.add_argument("--exts", nargs="*", default=sorted(_DEFAULT_EXTS))
    p.add_argument("--claude-model", default="anthropic/claude-3-5-sonnet-latest")
    p.add_argument("--gpt-model", default="openai/gpt-4o")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
