"""Dedup-gate + stratified sampler for the distillation training input.

Reads ``data/staged/MANIFEST.json`` (produced by the staging step) and the
uniform line-oriented ``.txt`` corpora it points at, then produces a
*deduplicated, trash-filtered, format-balanced* pool of chunks to distill.

Pipeline (per corpus, then across corpora):

  staged .txt
    │  stream lines
    ▼
  template-key dedup ── keep ONE real line per structural template
    │  (mirrors what the Rust runtime dedup removes before the pruner runs;
    │   <RAND>-masked so 45k cicd rows collapse to their few real templates)
    ▼
  trash gate ── drop lines that are nothing but a random blob (no signal)
    │
    ▼
  chunk (~max_tokens, log mode) ── per-corpus chunk pool
    │
    ▼
  water-fill across corpora ── balanced, capped allocation to a target count
    │  (no single format dominates by sheer volume)
    ▼
  deterministic select ── hash-ordered, reproducible, file-order-unbiased
    │
    ▼
  sampled chunks JSONL  ──►  run_distill --in <sampled.jsonl> consumes this

The sampler is corpus-agnostic and depends only on the manifest + staged text,
so it is fully unit-testable on tiny fixtures with no network and no teacher
calls. It performs NO paid API work — that is the distillation step that
consumes its output.

Measured dedup behaviour on the real corpora (full files, 2026-06-05):
  distsys_synth  100,000 rows -> 52 templates       (100% collapse: genuinely
                                                      only 52 distinct templates)
  cicd_failures   45,000 rows -> 44,517 templates    (1%: keys differ on real
                                                      categoricals — severity ×
                                                      branch × os × cloud × stage
                                                      × failure_type — not noise)
  api_failures   220,000 rows -> 220,000 templates   (0%: a 6-char random URL
                                                      slug per row; see note)

Two mechanisms with distinct jobs, so don't conflate them:
  * Template dedup removes redundancy WHERE IT EXISTS (distsys collapses to 52).
    It masks the long blobs/ids/numbers but keeps legitimate categorical
    diversity — correct.
  * The water-fill `max_share` cap is the DOMINATION guard. Short (<12 char)
    random ids (the api endpoint slug `lljugd`, short hashes) cannot be told
    apart from real short words (`Sphinx`, `GitLab`) by entropy without a
    dictionary, and flagging them risks dropping real identifiers — so they are
    intentionally left alone, and the cap bounds such corpora's contribution
    instead. Each corpus contributes up to its real unique pool, capped; the
    surplus from a thin corpus (distsys) redistributes to richer ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import chunk as chunk_text
from .normalize import is_low_signal, template_key_cached


@dataclass
class CorpusStats:
    name: str
    format: str
    raw_lines: int = 0
    unique_templates: int = 0
    dropped_trash: int = 0
    chunk_pool: int = 0
    selected: int = 0


@dataclass
class SampleSummary:
    target: int
    total_selected: int
    max_tokens: int
    max_share: float
    min_ratio: float
    per_corpus: list[CorpusStats] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


# --- path resolution ----------------------------------------------------------
def _resolve(path_str: str, *, root: Path, manifest_dir: Path) -> list[Path]:
    """Resolve a manifest path/glob into concrete files.

    Tries (in order) absolute, relative-to-root, relative-to-manifest-dir, and
    the bare basename under the manifest dir — so a manifest written with
    repo-root-relative paths works whether the sampler runs from the repo root
    or elsewhere. Supports glob patterns (``*``).
    """
    candidates: list[Path] = []
    p = Path(path_str)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)
        candidates.append(manifest_dir / p)
        candidates.append(manifest_dir / p.name)

    is_glob = any(ch in path_str for ch in "*?[")
    for c in candidates:
        if is_glob:
            base = c.parent
            matches = sorted(base.glob(c.name)) if base.exists() else []
            if matches:
                return matches
        elif c.exists():
            return [c]
    return []


def _corpus_files(entry: dict, *, root: Path, manifest_dir: Path) -> list[Path]:
    for key in ("staged_path", "source_glob", "source"):
        val = entry.get(key)
        if not val:
            continue
        files = _resolve(val, root=root, manifest_dir=manifest_dir)
        if files:
            return files
    return []


def _iter_lines(files: list[Path]) -> Iterator[str]:
    for f in files:
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    yield line.rstrip("\n")
        except OSError:
            continue


# --- dedup + trash gate -------------------------------------------------------
def dedup_and_gate(
    lines: Iterator[str], *, min_ratio: float = 0.30
) -> tuple[list[str], int, int, int]:
    """Collapse lines to one representative per template and drop trash.

    Returns ``(representatives, raw_lines, unique_templates, dropped_trash)``.
    Representative order follows first-seen order of each template (stable), so
    the result is deterministic for a given input stream.
    """
    seen: set[str] = set()
    reps: list[str] = []
    raw = 0
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        raw += 1
        key = template_key_cached(line)
        if key in seen:
            continue
        seen.add(key)
        if is_low_signal(line, min_ratio=min_ratio):
            dropped += 1
            continue
        reps.append(line)
    return reps, raw, len(seen), dropped


def chunk_representatives(reps: list[str], *, max_tokens: int) -> list[str]:
    """Pack representative lines into ~max_tokens log-mode chunks.

    Representatives are already template-distinct, so chunks are information-dense
    — which matches the runtime input distribution (the deterministic dedup has
    already stripped consecutive redundancy before the neural pruner sees text).
    """
    if not reps:
        return []
    return chunk_text("\n".join(reps), max_tokens=max_tokens, mode="log")


# --- stratified allocation ----------------------------------------------------
def water_fill(pools: dict[str, int], target: int, *, max_share: float = 0.25) -> dict[str, int]:
    """Max-min fair allocation of ``target`` slots across corpora.

    Every corpus gets an equal share; a corpus with fewer chunks than its share
    takes all it has and the surplus redistributes to corpora that still have
    headroom — repeated until the target is met or every pool is exhausted. A
    per-corpus ``max_share`` ceiling bounds how much any single format can
    contribute, so no corpus dominates even when others are small.

    Deterministic: ties broken by descending headroom then corpus name.
    """
    if target <= 0 or not pools:
        return {k: 0 for k in pools}
    cap = max(1, int(max_share * target))
    # Effective ceiling per corpus: its real pool, but never more than the cap.
    ceil = {k: min(v, cap) for k, v in pools.items()}
    alloc = {k: 0 for k in pools}
    remaining = min(target, sum(ceil.values()))
    active = {k for k, v in ceil.items() if v > 0}

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Hand out the final remainder one slot at a time, most-headroom first.
            order = sorted(active, key=lambda k: (-(ceil[k] - alloc[k]), k))
            for k in order:
                if remaining == 0:
                    break
                if alloc[k] < ceil[k]:
                    alloc[k] += 1
                    remaining -= 1
            break
        progressed = False
        for k in sorted(active):
            headroom = ceil[k] - alloc[k]
            take = min(share, headroom)
            if take > 0:
                alloc[k] += take
                remaining -= take
                progressed = True
            if alloc[k] >= ceil[k]:
                active.discard(k)
        if not progressed:
            break
    return alloc


def _stable_order(chunks: list[str]) -> list[int]:
    """Indices of ``chunks`` ordered by a stable content hash.

    Reproducible across runs/machines (sha1, not the salted builtin ``hash``)
    and not biased by file order, so sampling a prefix gives a spread-out subset.
    """
    keyed = [
        (hashlib.sha1(c.encode("utf-8")).hexdigest(), i) for i, c in enumerate(chunks)
    ]
    keyed.sort()
    return [i for _, i in keyed]


def select_chunks(chunks: list[str], k: int) -> list[str]:
    """Deterministically select ``k`` chunks from ``chunks`` (hash-ordered)."""
    if k >= len(chunks):
        return list(chunks)
    order = _stable_order(chunks)
    chosen = sorted(order[:k])  # restore original order among the chosen
    return [chunks[i] for i in chosen]


# --- top-level build ----------------------------------------------------------
def build_sample(
    manifest_path: Path,
    out_path: Path,
    *,
    target: int = 30000,
    max_tokens: int = 512,
    max_share: float = 0.25,
    min_ratio: float = 0.30,
    root: Path | None = None,
) -> SampleSummary:
    """Build the sampled distillation input JSONL from a staged manifest.

    Each output record is ``{"corpus", "src_path", "chunk_id", "text"}`` so the
    distill loader can treat it as a stream of ``(text, src, idx)`` items.
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    root = Path(root) if root is not None else Path.cwd()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("MANIFEST.json must be a JSON list of corpus objects")

    stats: list[CorpusStats] = []
    pools: dict[str, list[str]] = {}
    src_paths: dict[str, str] = {}

    for entry in manifest:
        name = entry.get("name") or "unknown"
        fmt = entry.get("format") or "unknown"
        files = _corpus_files(entry, root=root, manifest_dir=manifest_dir)
        st = CorpusStats(name=name, format=fmt)
        if not files:
            print(f"[warn] corpus {name!r}: no readable source files; skipping", file=sys.stderr)
            stats.append(st)
            continue
        reps, raw, uniq, dropped = dedup_and_gate(_iter_lines(files), min_ratio=min_ratio)
        chunks = chunk_representatives(reps, max_tokens=max_tokens)
        st.raw_lines = raw
        st.unique_templates = uniq
        st.dropped_trash = dropped
        st.chunk_pool = len(chunks)
        stats.append(st)
        if chunks:
            pools[name] = chunks
            src_paths[name] = str(files[0])

    alloc = water_fill({k: len(v) for k, v in pools.items()}, target, max_share=max_share)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    by_name = {s.name: s for s in stats}
    with out_path.open("w", encoding="utf-8") as fh:
        for name in sorted(pools):
            chosen = select_chunks(pools[name], alloc.get(name, 0))
            by_name[name].selected = len(chosen)
            for idx, text in enumerate(chosen):
                fh.write(
                    json.dumps(
                        {
                            "corpus": name,
                            "src_path": src_paths[name],
                            "chunk_id": idx,
                            "text": text,
                        }
                    )
                    + "\n"
                )
                total += 1

    summary = SampleSummary(
        target=target,
        total_selected=total,
        max_tokens=max_tokens,
        max_share=max_share,
        min_ratio=min_ratio,
        per_corpus=stats,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dedup + trash-filter + stratified-sample staged corpora into a distill input JSONL."
    )
    p.add_argument(
        "--manifest",
        default="data/staged/MANIFEST.json",
        help="path to the staged MANIFEST.json (default: data/staged/MANIFEST.json)",
    )
    p.add_argument(
        "--out",
        default="data/sampled/v0_input.jsonl",
        help="output sampled-chunk JSONL (default: data/sampled/v0_input.jsonl)",
    )
    p.add_argument("--target", type=int, default=30000, help="target number of chunks")
    p.add_argument("--max-tokens", type=int, default=512, help="per-chunk token cap (cl100k)")
    p.add_argument(
        "--max-share",
        type=float,
        default=0.25,
        help="max fraction of the sample any single corpus may contribute",
    )
    p.add_argument(
        "--min-signal-ratio",
        type=float,
        default=0.30,
        dest="min_ratio",
        help="drop lines whose word-like char fraction is below this (trash gate)",
    )
    p.add_argument("--root", default=None, help="repo root for resolving manifest paths (default: cwd)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"manifest not found: {manifest_path}\n"
            "The staging step (see ml_pipeline/COMPOSER_TASK.md) must produce "
            "data/staged/MANIFEST.json + the staged .txt corpora first.",
            file=sys.stderr,
        )
        return 1
    summary = build_sample(
        manifest_path,
        Path(args.out),
        target=args.target,
        max_tokens=args.max_tokens,
        max_share=args.max_share,
        min_ratio=args.min_ratio,
        root=Path(args.root) if args.root else None,
    )
    print(summary.to_json())
    print(
        f"\nwrote {summary.total_selected} chunks to {args.out} "
        f"(target {summary.target})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
