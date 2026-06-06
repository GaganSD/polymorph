"""(log, question, answer) triple generation for the answer-survival benchmark.

The benchmark asks the real question a log compressor must answer: *after* you
compress a chunk of logs, can a downstream reader still recover the fact that
mattered? We operationalize "the fact that mattered" as an **answer needle** — a
rare, salient substring (an exception type, an HTTP status, a request id, an
incident number, a client IP, a severity) extracted from the chunk by regex,
paired with a templated extraction question.

A triple is built so the answer occurs (ideally) **once** in the chunk, so
"survival" is an unambiguous test: did the compressed output still contain it?

This is deliberately GPU-free and deterministic: extraction is pure regex, no
model, no API. Triples can be auto-mined from any corpus of log text
(``build_triples_from_paths``) or taken from the small curated fixture set
(``CURATED_TRIPLES``) the tests use so they never depend on the gitignored
``data/`` tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AnswerTriple:
    doc_id: str
    text: str          # the (uncompressed) log chunk
    question: str      # extraction query
    answer: str        # the exact needle that must survive
    fact_type: str     # e.g. "exception", "http_status", "request_id"
    source: str        # corpus / file the chunk came from


# ---------------------------------------------------------------------------
# Extractors — each yields (fact_type, question, answer) candidates from a chunk.
# Higher-priority fact types are listed first; the builder prefers a candidate
# whose answer is rare (unique) in the chunk.
# ---------------------------------------------------------------------------

# (fact_type, compiled regex, capture-group index, question template)
_EXTRACTORS: list[tuple[str, re.Pattern, int, str]] = [
    # Python / Java exception type, e.g. "ImportError: ...", "ValueError: ...".
    ("exception", re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b"), 1,
     "Which exception type was raised?"),
    # ServiceNow-style incident id.
    ("incident", re.compile(r"\b(INC\d{4,})\b"), 1,
     "What incident number is referenced?"),
    # request_id=<token>
    ("request_id", re.compile(r"request_id[=:]\s*([A-Za-z0-9_-]+)"), 1,
     "What was the request_id of the affected request?"),
    # HTTP 4xx/5xx status code (near 'status'/'HTTP'/'code', or standalone 5xx).
    ("http_status", re.compile(r"\b(?:status|HTTP|code)[=:\s]+([45]\d{2})\b", re.IGNORECASE), 1,
     "What HTTP status code was returned?"),
    # errno / error code: errno=NN or "error code 0xABCD".
    ("error_code", re.compile(r"\berrno[=:]\s*(\d+)\b|\berror code\s+(0x[0-9A-Fa-f]+|\d+)\b"), 0,
     "What error code was reported?"),
    # client_ip=a.b.c.d (prefer the labeled one over any bare IP).
    ("client_ip", re.compile(r"client_ip[=:]\s*(\d{1,3}(?:\.\d{1,3}){3})"), 1,
     "What client IP is associated with the event?"),
    # Highest severity keyword present.
    ("severity", re.compile(r"\b(FATAL|CRITICAL|ERROR)\b"), 1,
     "What is the most severe log level present in this chunk?"),
]


def _candidates(text: str) -> list[tuple[str, str, str]]:
    """Return (fact_type, question, answer) candidates, in priority order."""
    out: list[tuple[str, str, str]] = []
    for fact_type, pat, grp, question in _EXTRACTORS:
        for m in pat.finditer(text):
            if grp == 0:
                # alternation: take the first non-None group
                answer = next((g for g in m.groups() if g), None)
            else:
                answer = m.group(grp)
            if answer and len(answer) >= 2:
                out.append((fact_type, question, answer))
    return out


def _best_triple_for_chunk(doc_id: str, text: str, source: str) -> AnswerTriple | None:
    """Pick the highest-priority candidate whose answer is UNIQUE in the chunk."""
    seen_types: set[str] = set()
    for fact_type, question, answer in _candidates(text):
        if fact_type in seen_types:
            continue
        seen_types.add(fact_type)
        # Prefer a needle that occurs exactly once: unambiguous survival test.
        if text.count(answer) == 1:
            return AnswerTriple(doc_id, text, question, answer, fact_type, source)
    # Fall back to the first candidate even if it repeats (still a valid test).
    cands = _candidates(text)
    if cands:
        fact_type, question, answer = cands[0]
        return AnswerTriple(doc_id, text, question, answer, fact_type, source)
    return None


def _line_windows(lines: list[str], window: int, stride: int) -> Iterable[list[str]]:
    n = len(lines)
    if n <= window:
        if lines:
            yield lines
        return
    i = 0
    while i < n:
        chunk = lines[i : i + window]
        if chunk:
            yield chunk
        if i + window >= n:
            return
        i += stride


def build_triples_from_text(
    text: str, source: str, window_lines: int = 40, stride: int | None = None, max_chunks: int | None = None
) -> list[AnswerTriple]:
    """Mine answer triples from a block of log text by line-windowing."""
    stride = stride or window_lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    triples: list[AnswerTriple] = []
    for ci, chunk_lines in enumerate(_line_windows(lines, window_lines, stride)):
        if max_chunks is not None and ci >= max_chunks:
            break
        chunk = "\n".join(chunk_lines)
        t = _best_triple_for_chunk(f"{source}#{ci}", chunk, source)
        if t is not None:
            triples.append(t)
    return triples


def build_triples_from_paths(
    paths: Iterable[Path],
    window_lines: int = 40,
    max_per_file: int = 5,
    max_total: int | None = None,
    max_bytes_per_file: int = 2_000_000,
) -> list[AnswerTriple]:
    """Mine answer triples from a set of log files (``.log``/``.txt``/``.json``).

    Reads at most ``max_bytes_per_file`` per file and emits up to ``max_per_file``
    triples each, capped at ``max_total`` overall. Files that can't be read as
    UTF-8 are skipped.
    """
    triples: list[AnswerTriple] = []
    for path in paths:
        if max_total is not None and len(triples) >= max_total:
            break
        try:
            text = path.read_text(errors="ignore")[:max_bytes_per_file]
        except (OSError, UnicodeError):
            continue
        got = build_triples_from_text(text, source=path.name, window_lines=window_lines)
        triples.extend(got[:max_per_file])
    if max_total is not None:
        triples = triples[:max_total]
    return triples


def collect_log_files(root: Path) -> list[Path]:
    """Recursively collect log-like files under ``root`` (skipping sidecars)."""
    exts = {".log", ".txt", ".json", ".jsonl"}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            name = p.name
            if name.endswith(".survive") or name.startswith("potentialAnomalies"):
                continue
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Curated fixtures — small, self-contained triples for tests (no data/ dep).
# ---------------------------------------------------------------------------

_FIXTURE_DOCS: list[tuple[str, str]] = [
    (
        "distsys",
        "\n".join(
            f"2023-11-20T08:40:5{i%10}.000 INFO [ServiceA] heartbeat ok request_id=10{i:02d} client_ip=10.0.0.{i%5}"
            for i in range(30)
        )
        + "\n2023-11-20T08:41:02.111 FATAL [ServiceB] Crash request_id=99042 client_ip=192.168.1.185 time_taken=72ms",
    ),
    (
        "traceback",
        "Traceback (most recent call last): File \"app.py\", line 42, in run\n"
        "  File \"db.py\", line 10, in connect\n"
        "ConnectionResetError: connection reset by peer",
    ),
    (
        "servicenow",
        "\n".join(
            f"29/2/2016 0{i}:23 incident=INC000004{i} state=New priority=3 category=Category 55"
            for i in range(5)
        ),
    ),
    (
        "api",
        "\n".join(
            f"GET /v1/items 200 ok latency=12ms id=req{i}" for i in range(20)
        )
        + "\nPOST /v1/checkout HTTP status=503 service unavailable id=reqX9 retries=3",
    ),
]


def curated_triples() -> list[AnswerTriple]:
    """Deterministic, dependency-free triples for tests."""
    out: list[AnswerTriple] = []
    for source, text in _FIXTURE_DOCS:
        t = _best_triple_for_chunk(f"{source}#0", text, source)
        if t is not None:
            out.append(t)
    return out
