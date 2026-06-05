"""One-command post-distill pipeline: raw distilled JSONL -> train/val shards.

Chains the existing library steps into a single CLI so a distilled corpus can be
turned into trainer-ready shards without hand-wiring QC, labeling, and splitting:

    lamr-pipeline --distilled <in.jsonl> --out-dir <dir> \
        [--val-frac 0.05] [--seed 42] [--lang-detect] [--min-tokens N]

Steps:
  1. Load distilled JSONL (same robust line-skipping as
     ``distill.run_distill._load_sampled``: skip blanks + empty original/compressed).
  2. QC-filter via ``qc.filter.filter_records`` (drops hallucinated/low-quality pairs).
  3. Per surviving pair: derive a cl100k keep/drop mask (``label.align.derive_mask``)
     and AST hop-decay soft weights (``label.ast_split.split_labels``). Language is
     inferred from ``src_path`` (.py -> python, .json -> json, else None) unless
     ``--lang-detect`` is off.
  4. Deterministic, leak-free train/val split: hash the *source identity*
     (``src_path``, not ``chunk_id``) with hashlib so every chunk of one source lands
     in the same split and the assignment is stable across runs/interpreters.
  5. Write ``<out-dir>/train.jsonl`` and ``<out-dir>/val.jsonl`` in the exact shard
     schema ``train.dataset.LabeledShardDataset`` consumes:
        {input_ids, tags(0=keep,1=drop), w_semantic, w_dependency, is_code, src_path}

Pure stdlib + already-declared deps (tiktoken/tree-sitter live behind the label API).
No network, no Bedrock, no cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .label.align import derive_mask
from .label.ast_split import split_labels
from .qc.filter import filter_records
from .qc.metrics import QCRecord

# Number of leading sha256 hex digits used to bucket a source into train/val.
# 8 hex digits = 32 bits of resolution, plenty for a fractional split.
_SPLIT_HEX_WIDTH = 8


@dataclass
class DistilledPair:
    """A loaded distilled record carrying the fields the pipeline needs."""

    original: str
    compressed: str
    src_path: str


def _load_distilled(path: Path) -> list[DistilledPair]:
    """Load distilled JSONL, skipping blank lines and empty original/compressed.

    Mirrors the robust line-skipping of ``run_distill._load_sampled`` so a
    partially-written / crash-truncated build file degrades gracefully.
    """
    pairs: list[DistilledPair] = []
    bad_lines = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash-truncated / partially-written build file can leave a
                # malformed final line. Skip it (the docstring promises graceful
                # degradation) rather than aborting the whole run.
                bad_lines += 1
                continue
            original = rec.get("original")
            compressed = rec.get("compressed")
            if not original or not compressed:
                continue
            src_path = rec.get("src_path") or ""
            pairs.append(
                DistilledPair(
                    original=original,
                    compressed=compressed,
                    src_path=src_path,
                )
            )
    if bad_lines:
        print(f"[warn] skipped {bad_lines} malformed JSON line(s) in {path}", file=sys.stderr)
    return pairs


def _infer_lang(src_path: str) -> str | None:
    """Infer the AST language from a source path / corpus tag.

    ``src_path`` may be ``corpus:path`` (sampler convention) — we key off the
    actual file suffix. .py -> python, .json/.jsonl -> json, else None (prose).
    """
    if not src_path:
        return None
    # Sampler encodes source as ``corpus:src_path``; take the path component.
    path_part = src_path.split(":", 1)[-1] if ":" in src_path else src_path
    suffix = Path(path_part).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in (".json", ".jsonl"):
        return "json"
    return None


def _split_bucket(src_path: str, val_frac: float, seed: int) -> str:
    """Deterministically assign a *source* to "train" or "val".

    Hashes ``src_path`` (the corpus/file identity, NOT chunk_id) so every chunk of
    one source lands in the same split — no leakage. Uses hashlib (stable across
    runs and interpreters), not Python's salted ``hash()``.
    """
    if val_frac <= 0.0:
        return "train"
    key = f"{seed}:{src_path}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Map the leading HEX_WIDTH hex digits to a fraction in [0, 1). The divisor
    # is derived from the slice width so the two can't drift out of lockstep.
    bucket = int(digest[:_SPLIT_HEX_WIDTH], 16) / float(1 << (4 * _SPLIT_HEX_WIDTH))
    return "val" if bucket < val_frac else "train"


def _build_shard_record(pair: DistilledPair, lang: str | None) -> dict | None:
    """Label one pair into a shard line. Returns None if it has no tokens."""
    align = derive_mask(pair.original, pair.compressed)
    if not align.token_ids:
        return None
    split = split_labels(
        pair.original,
        align.keep_mask,
        align.spans,
        lang=lang,
    )
    # Tag convention (see train/dataset.py + run_e2e_smoke.sh): 0 = keep, 1 = drop.
    tags = [0 if k else 1 for k in split.keep_mask]
    return {
        "input_ids": align.token_ids,
        "tags": tags,
        "w_semantic": split.w_semantic,
        "w_dependency": split.w_dependency,
        "is_code": split.is_code,
        "src_path": pair.src_path,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lamr-pipeline",
        description="Turn a raw distilled JSONL into train/val labeled shards "
        "(QC filter -> cl100k mask + AST weights -> leak-free split).",
    )
    p.add_argument("--distilled", required=True, help="input distilled JSONL")
    p.add_argument("--out-dir", required=True, help="output dir for train.jsonl + val.jsonl")
    p.add_argument("--val-frac", type=float, default=0.05, help="validation fraction (default 0.05)")
    p.add_argument("--seed", type=int, default=42, help="hash seed for the split (default 42)")
    p.add_argument(
        "--lang-detect",
        dest="lang_detect",
        action="store_true",
        default=True,
        help="infer AST language from src_path suffix (default on)",
    )
    p.add_argument(
        "--no-lang-detect",
        dest="lang_detect",
        action="store_false",
        help="treat every record as prose (skip the AST walk)",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="drop survivors whose original tokenizes to fewer than N tokens",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    in_path = Path(args.distilled)
    if not in_path.is_file():
        print(f"distilled file not found: {in_path}", file=sys.stderr)
        return 1

    pairs = _load_distilled(in_path)
    total_in = len(pairs)
    if not pairs:
        print("no usable distilled records (all blank/empty)", file=sys.stderr)
        return 1

    # QC filter. Compute metrics fresh (don't trust any qc{} field in the input,
    # which may be stale or absent on a partially-written build). qc_records[i]
    # corresponds to pairs[i] positionally; filter_records returns a subset of
    # those same objects, so we recover the surviving pairs by id-set membership
    # over the aligned zip (no fragile id->pair dict, no duplicate collapse).
    qc_records = [QCRecord.compute(p.original, p.compressed) for p in pairs]
    survivors, report = filter_records(qc_records)
    survivor_ids = {id(r) for r in survivors}
    surviving_pairs = [p for rec, p in zip(qc_records, pairs) if id(rec) in survivor_ids]
    print(
        f"[qc] total={report.get('total', total_in)} "
        f"after_hard_floor={report.get('after_hard_floor', '-')} "
        f"after_vr_filter={report.get('after_vr_filter', '-')} "
        f"kept={report.get('kept', len(survivors))} "
        f"vr_hard_floor={report.get('vr_hard_floor', '-')} "
        f"vr_cutoff={report.get('vr_cutoff', '-')} "
        f"ag_cutoff={report.get('ag_cutoff', '-')}"
    )
    dropped = total_in - len(survivors)
    print(f"[qc] dropped {dropped} of {total_in} pairs ({len(survivors)} survived)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    counts = {"train": 0, "val": 0}
    code_count = 0
    token_total = 0
    emitted = 0
    skipped_short = 0

    with train_path.open("w", encoding="utf-8") as train_fh, val_path.open(
        "w", encoding="utf-8"
    ) as val_fh:
        handles = {"train": train_fh, "val": val_fh}
        for pair in surviving_pairs:
            lang = _infer_lang(pair.src_path) if args.lang_detect else None
            shard = _build_shard_record(pair, lang)
            if shard is None:
                continue
            if args.min_tokens and len(shard["input_ids"]) < args.min_tokens:
                skipped_short += 1
                continue
            bucket = _split_bucket(pair.src_path, args.val_frac, args.seed)
            handles[bucket].write(json.dumps(shard) + "\n")
            counts[bucket] += 1
            emitted += 1
            if shard["is_code"]:
                code_count += 1
            token_total += len(shard["input_ids"])

    pct_code = (100.0 * code_count / emitted) if emitted else 0.0
    mean_tokens = (token_total / emitted) if emitted else 0.0
    print("[summary]")
    print(f"  total in           : {total_in}")
    print(f"  QC-survived        : {len(survivors)}")
    if args.min_tokens:
        print(f"  dropped < {args.min_tokens} tokens : {skipped_short}")
    print(f"  emitted            : {emitted}")
    print(f"  train / val        : {counts['train']} / {counts['val']}")
    print(f"  code / prose       : {pct_code:.1f}% code, {100.0 - pct_code:.1f}% prose")
    print(f"  mean tokens/record : {mean_tokens:.1f}")
    print(f"  wrote {train_path}")
    print(f"  wrote {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
