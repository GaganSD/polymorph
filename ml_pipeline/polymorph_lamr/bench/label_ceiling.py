"""Phase 0b: label-ceiling QA — the $0 kill gate for the answer-survival SOTA bet.

The model can only ever be as good as its labels. Every training label comes from
the teacher's compressed text (``derive_mask(original, compressed)`` keeps the
tokens the teacher kept). So if the teacher already dropped an answer needle that
``keep-severity`` keeps at the *same* compression, NO model trained on these
labels can beat keep-severity — the ceiling is below the baseline, and spending
GPU credit is pointless until the labels improve.

This module measures, over the distilled records, on chunks that contain a unique
answer needle (mined by the existing benchmark extractors):

  teacher-ceiling : needle survives the teacher's ``compressed`` text?
  label-ceiling   : needle survives the cl100k keep-mask the trainer learns
                    (``label.align.derive_mask``)?
  keep-severity   : needle survives keep-severity at the teacher's *achieved*
                    drop rate on the same original (iso-ratio baseline)?

Verdict: if teacher-ceiling survival < keep-severity survival, the labels cap the
model below the baseline -> STOP and fix labels before any training spend.

GPU-free, deterministic, no network. Reuses the benchmark's needle extractors and
survival test so "needle" means exactly what the benchmark means by it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from ..label.align import derive_mask
from .methods import KeepSeverityHeuristic, token_count
from .survival import answer_survives
from .triples import _best_semantic_triple_for_chunk, _best_triple_for_chunk


@dataclass
class CeilingCounts:
    chunks: int = 0            # records seen
    with_needle: int = 0       # records with a mineable needle
    needle_in_original: int = 0  # sanity: needle actually present in original
    teacher_survived: int = 0  # needle survives the teacher compressed text
    label_survived: int = 0    # needle survives the derived keep-mask
    keepsev_survived: int = 0  # needle survives keep-severity at teacher's ratio
    sum_teacher_drop: float = 0.0  # for mean achieved drop reporting
    # Per-fact-type breakdown: fact_type -> [denom, teacher_survived, keepsev_survived]
    by_type: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )


def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _kept_text(original: str, compressed: str) -> tuple[str, float]:
    """Reconstruct the text the trainer's keep-mask preserves, and the teacher's
    achieved token drop rate. ``derive_mask`` is the exact label source."""
    align = derive_mask(original, compressed)
    enc = _enc()
    kept_ids = [tid for tid, keep in zip(align.token_ids, align.keep_mask) if keep]
    n = len(align.token_ids)
    drop = 1.0 - (len(kept_ids) / n) if n else 0.0
    return enc.decode(kept_ids), drop


def iter_records(path: Path, limit: int | None = None):
    """Yield (original, compressed, src_path) from a distilled JSONL, skipping
    blanks / malformed / empty-pair lines (mirrors the pipeline loader)."""
    seen = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = rec.get("original")
            compressed = rec.get("compressed")
            if not original or not compressed:
                continue
            yield original, compressed, rec.get("src_path") or ""
            seen += 1
            if limit is not None and seen >= limit:
                return


def measure(path: Path, limit: int | None = None, semantic: bool = False) -> CeilingCounts:
    c = CeilingCounts()
    keepsev = KeepSeverityHeuristic()
    extract = _best_semantic_triple_for_chunk if semantic else _best_triple_for_chunk
    for ci, (original, compressed, source) in enumerate(iter_records(path, limit)):
        c.chunks += 1
        triple = extract(f"{source}#{ci}", original, source)
        if triple is None:
            continue
        c.with_needle += 1
        needle = triple.answer
        if not answer_survives(needle, original):
            # Extractor pulled a needle that doesn't substring-match (rare; e.g.
            # regex normalization). Skip — it can't be a fair survival test.
            continue
        c.needle_in_original += 1
        bucket = c.by_type[triple.fact_type]
        bucket[0] += 1

        # Teacher ceiling: did the teacher keep the needle at all?
        if answer_survives(needle, compressed):
            c.teacher_survived += 1
            bucket[1] += 1

        # Label ceiling: does the derived keep-mask preserve the needle?
        kept_text, teacher_drop = _kept_text(original, compressed)
        c.sum_teacher_drop += teacher_drop
        if answer_survives(needle, kept_text):
            c.label_survived += 1

        # Iso-ratio baseline: keep-severity at the teacher's achieved drop rate.
        r = max(0.0, min(1.0, teacher_drop))
        if answer_survives(needle, keepsev.compress(original, r)):
            c.keepsev_survived += 1
            bucket[2] += 1
    return c


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def format_report(c: CeilingCounts) -> str:
    den = c.needle_in_original
    mean_drop = (c.sum_teacher_drop / den) if den else 0.0
    lines = [
        "== LaMR label-ceiling QA (Phase 0b) ==",
        f"records scanned        : {c.chunks}",
        f"with mineable needle   : {c.with_needle}",
        f"needle present in orig : {c.needle_in_original}  (the survival denominator)",
        f"teacher mean drop rate : {mean_drop:.3f}",
        "",
        f"teacher-ceiling survival : {_pct(c.teacher_survived, den)}  "
        f"({c.teacher_survived}/{den})  <- the hard cap on any model",
        f"label-ceiling survival   : {_pct(c.label_survived, den)}  "
        f"({c.label_survived}/{den})  <- what the trainer actually learns",
        f"keep-severity @ same drop: {_pct(c.keepsev_survived, den)}  "
        f"({c.keepsev_survived}/{den})  <- the baseline to beat",
        "",
        "per-fact-type survival (teacher vs keep-severity, n):",
    ]
    for ftype, (n, tsurv, ksurv) in sorted(c.by_type.items(), key=lambda kv: -kv[1][0]):
        lines.append(
            f"  {ftype:<12} teacher {_pct(tsurv, n):>6}  keep-sev {_pct(ksurv, n):>6}  (n={n})"
        )
    lines.append("")
    if den:
        teacher = c.teacher_survived / den
        baseline = c.keepsev_survived / den
        if teacher < baseline - 1e-9:
            lines.append(
                "VERDICT: GATE FAILS. Teacher labels preserve FEWER needles than "
                "keep-severity at the same compression. No model trained on these "
                "labels can beat the baseline. Fix labels (2-teacher agreement / "
                "rule-augment severity) before spending GPU credit."
            )
        elif teacher < baseline + 1e-9:
            lines.append(
                "VERDICT: MARGINAL. Teacher labels only match keep-severity. The "
                "model's edge must come entirely from sub-line precision, not from "
                "better line selection. Proceed, but expect a thin win."
            )
        else:
            lines.append(
                "VERDICT: GATE CLEARS (label side). Teacher labels preserve MORE "
                f"needles than keep-severity (headroom "
                f"{100.0 * (teacher - baseline):.1f} pts). A model that fits these "
                "labels can beat the baseline; the bet is live."
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lamr-label-ceiling",
        description="Phase 0b: does the teacher label preserve answer needles as "
        "well as keep-severity at the same compression? The $0 SOTA kill gate.",
    )
    p.add_argument("--distilled", required=True, help="distilled JSONL (original+compressed)")
    p.add_argument("--limit", type=int, default=None, help="cap records scanned")
    p.add_argument("--semantic", action="store_true", help="mine SEMANTIC free-text needles instead of structural")
    p.add_argument("--out", type=Path, default=None, help="write counts as JSON here")
    args = p.parse_args(argv)

    in_path = Path(args.distilled)
    if not in_path.is_file():
        print(f"distilled file not found: {in_path}", file=sys.stderr)
        return 1

    c = measure(in_path, args.limit, semantic=args.semantic)
    report = format_report(c)
    print(report)
    if args.out:
        payload = {k: v for k, v in c.__dict__.items() if k != "by_type"}
        payload["by_type"] = {
            ft: {"n": n, "teacher_survived": t, "keepsev_survived": k}
            for ft, (n, t, k) in c.by_type.items()
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
