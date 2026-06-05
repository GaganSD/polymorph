"""CLI: distill a directory tree of logs/traces into JSONL training pairs.

Primary path is the multi-teacher ENSEMBLE (E3): each chunk is compressed by
several teachers and the best-QC output is kept as the training target. Default
teachers are two AWS Bedrock open-weight models (see `providers.py`):

* deepseek-v32  via AWS Bedrock  -> AWS credential chain + AWS_REGION
* minimax-m21   via AWS Bedrock  -> AWS credential chain + AWS_REGION

    AWS_REGION=eu-north-1 \
        python -m polymorph_lamr.distill.run_distill \
        --sampled data/sampled/v0_input.jsonl --out data/distilled/v0.jsonl --concurrency 8

A teacher that errors (or is throttled) is dropped from per-chunk best-QC
selection rather than failing the run. A legacy two-teacher (Claude + GPT-4o)
mode is kept for back-compat: --mode pair.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


def _load_sampled(path: Path) -> list[tuple[str, str, int]]:
    """Load pre-chunked items from a sampler JSONL (see ``distill.sampler``).

    Each line is ``{"corpus","src_path","chunk_id","text"}``. The sampler has
    already deduped, trash-filtered, chunked and format-balanced the corpora, so
    its chunks feed straight through as ``(text, src, idx)`` — no file walking or
    re-chunking. ``src`` carries the corpus name so per-chunk warnings/output stay
    traceable to a format.
    """
    items: list[tuple[str, str, int]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if not text:
                continue
            corpus = rec.get("corpus") or ""
            src_path = rec.get("src_path") or corpus or str(path)
            src = f"{corpus}:{src_path}" if corpus else src_path
            items.append((text, src, int(rec.get("chunk_id", 0))))
    return items


def _parse_teachers(specs: list[str] | None) -> list[TeacherSpec]:
    """Parse `--teachers name=spec` entries into routed TeacherSpecs.

    The spec is provider-prefixed and routed via `TeacherSpec.from_spec`, e.g.
    `--teachers deepseek-v32=bedrock/deepseek.v3.2 minimax-m21=bedrock/minimax.minimax-m2.1`.
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
    if args.sampled:
        items = _load_sampled(Path(args.sampled))
        source_desc = f"sampler JSONL {args.sampled}"
    elif args.input:
        files = _iter_files(Path(args.input), set(args.exts))
        items = _load_items(files, max_tokens=args.max_tokens)
        source_desc = f"{len(files)} files"
    else:
        print("provide either --sampled <jsonl> or --in <dir/file>", file=sys.stderr)
        return 1
    if not items:
        print("no items found", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        # Cap chunks for smoke runs (cheap live-path validation before a full
        # build). Deterministic prefix — items are file/chunk-ordered.
        items = items[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_cost = 0.0

    # Stream results to disk incrementally (flush every 25) so a long, paid build
    # is crash-safe: a failure loses at most the last unflushed batch, not hours.
    with out_path.open("w", encoding="utf-8") as out_fh:
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
                f"distilling {len(items)} chunks from {source_desc} via "
                f"{len(teachers)} teachers "
                f"({', '.join(t.name for t in teachers)}); concurrency={args.concurrency}"
            )
            async for r in distill_ensemble_many(items, cfg=cfg, concurrency=args.concurrency):
                out_fh.write(r.to_json() + "\n")
                n += 1
                total_cost += r.cost_usd
                if r.errors:
                    print(f"[warn] {r.src_path}#{r.chunk_id}: {r.errors}", file=sys.stderr)
                if n % 25 == 0:
                    out_fh.flush()
                    print(f"  progress: {n}/{len(items)} cost=${total_cost:.4f}")
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
            print(f"distilling {len(items)} chunks (pair mode) from {source_desc}")
            async for r in distill_many(items, cfg=cfg, concurrency=args.concurrency):
                out_fh.write(r.to_json() + "\n")
                n += 1
                total_cost += r.cost_usd
                if r.errors:
                    print(f"[warn] {r.src_path}#{r.chunk_id} errors: {r.errors}", file=sys.stderr)

    print(f"wrote {n} records to {args.output} (cost ${total_cost:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distill logs/traces into compressed training pairs.")
    p.add_argument("--in", dest="input", default=None,
                   help="source file or directory (walked + chunked); omit when using --sampled")
    p.add_argument("--sampled", default=None,
                   help="consume a pre-chunked sampler JSONL ({corpus,src_path,chunk_id,text}) "
                        "from distill.sampler instead of walking --in")
    p.add_argument("--out", dest="output", required=True, help="output JSONL path")
    p.add_argument("--mode", choices=["ensemble", "pair"], default="ensemble",
                   help="ensemble = open-weight teachers (default); pair = legacy Claude+GPT-4o")
    p.add_argument("--teachers", nargs="*", default=None,
                   help="ensemble teachers as name=spec (default: deepseek-v32 + minimax-m21, both Bedrock)")
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
