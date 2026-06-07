"""Phase 0c: build a powered, held-out, semantically-representative benchmark.

Mines (log, question, answer) triples from the decoded **validation** shard — text
the model never trains on — so the benchmark measures generalization, not recall of
training chunks (closes the train/bench leakage risk, 0a). Each chunk contributes
up to two needles:

  * a STRUCTURAL needle (status code, IP, exception, id) — what the deterministic
    floor locks; floored methods survive these ~100% by construction.
  * a SEMANTIC needle (a free-text root-cause / resolution / message phrase) — what
    no regex can lock; only a model that learns salience preserves it under a tight
    budget. This is the class that actually discriminates models.

Reporting survival split by class makes the floor's structural saturation and the
open semantic gap both visible, so a SOTA claim rests on the semantic column.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import tiktoken

from .triples import AnswerTriple, _best_semantic_triple_for_chunk, _best_triple_for_chunk


def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def build_from_shard(shard_path: Path, source: str, limit: int | None = None) -> list[AnswerTriple]:
    enc = _enc()
    triples: list[AnswerTriple] = []
    seen_answers: set[str] = set()
    for ci, line in enumerate(shard_path.open(encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        if limit is not None and ci >= limit:
            break
        rec = json.loads(line)
        ids = rec.get("input_ids")
        if not ids:
            continue
        text = enc.decode(ids)
        for builder in (_best_semantic_triple_for_chunk, _best_triple_for_chunk):
            t = builder(f"{source}#{ci}", text, source)
            if t is None:
                continue
            # De-dup identical needles so a repetitive synthetic corpus can't
            # inflate the count with the same answer over and over.
            key = f"{t.fact_type}:{t.answer}"
            if key in seen_answers:
                continue
            seen_answers.add(key)
            triples.append(t)
    return triples


def class_counts(triples: list[AnswerTriple]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in triples:
        cls = "semantic" if t.fact_type.startswith("semantic:") else "structural"
        out[cls] = out.get(cls, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lamr-build-heldout",
        description="Mine a held-out structural+semantic benchmark from a val shard.",
    )
    p.add_argument("--val-shard", required=True, help="path to val.jsonl (input_ids shard)")
    p.add_argument("--source", default="val", help="source tag for the triples")
    p.add_argument("--limit", type=int, default=None, help="cap records scanned")
    p.add_argument("--out", required=True, type=Path, help="output triples JSON")
    args = p.parse_args(argv)

    triples = build_from_shard(Path(args.val_shard), args.source, args.limit)
    counts = class_counts(triples)
    payload = {
        "n": len(triples),
        "class_counts": counts,
        "fact_type_counts": _fact_counts(triples),
        "triples": [asdict(t) for t in triples],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"built {len(triples)} held-out triples: {counts}")
    print(f"fact types: {payload['fact_type_counts']}")
    print(f"wrote {args.out}")
    return 0


def _fact_counts(triples: list[AnswerTriple]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in triples:
        out[t.fact_type] = out.get(t.fact_type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def load_triples(path: Path) -> list[AnswerTriple]:
    payload = json.loads(Path(path).read_text())
    return [AnswerTriple(**d) for d in payload["triples"]]


if __name__ == "__main__":
    raise SystemExit(main())
