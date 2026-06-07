"""Deterministic structural locker — the decode-time insurance floor (Phase 0d).

A small set of regexes over atomic high-salience facts an operator almost always
needs in a log: severities, HTTP 4xx/5xx, error/errno codes, IPv4 addresses,
UUIDs, exception types, incident ids, request ids. The floor force-keeps every
cl100k token that overlaps a match, so no ranker — and no compression budget — can
drop them. The learned model stays free to trim everything else aggressively (its
sub-line-precision edge); it simply can never lose a structural atom. That is the
failure mode the answer-survival benchmark punishes and exactly where the
1-teacher labels were weakest (client_ip: teacher 25.7% vs keep-severity 68.6%).

HONESTY NOTE (read before trusting a benchmark number): these regexes overlap the
benchmark's needle extractors (`bench/triples.py`). So a floored method's survival
on those fact types is ~100% BY CONSTRUCTION — a deterministic guarantee, not a
measure of model generalization. Two consequences:
  * The model's *generalization* must be judged on non-structural / semantic
    needles (the LLM-judge variant, or held-out classes 0c adds). Do not read a
    floored method's structural-needle survival as "the model is good".
  * What the floor legitimately proves against keep-severity is mechanism, not
    magic: a TOKEN-level structural prior keeps atoms sitting on non-severe lines
    that a LINE-level severity prior discards. `random + floor` beating
    keep-severity is the clean, non-circular demonstration of that.
"""

from __future__ import annotations

import re

from ..label.align import encode_with_spans

# Atomic high-salience patterns. Full-match spans are locked (not capture groups).
# Kept deliberately close to the production auto-lock intent; parallels (but is not
# imported from) the benchmark's extractors so the two can diverge if needed.
_LOCK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b"),       # exception type
    re.compile(r"\b(?:FATAL|CRITICAL|ERROR|EXCEPTION|TRACEBACK|WARN(?:ING)?)\b"),  # severity
    re.compile(r"(?:status|HTTP|code)[=:\s]+[45]\d{2}\b", re.IGNORECASE),  # http 4xx/5xx
    re.compile(r"\berrno[=:]\s*\d+\b"),                              # errno
    re.compile(r"\berror code\s+(?:0x[0-9A-Fa-f]+|\d+)\b", re.IGNORECASE),  # error code
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),                     # IPv4
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),  # UUID
    re.compile(r"\bINC\d{4,}\b"),                                   # incident id
    re.compile(r"request_id[=:]\s*[A-Za-z0-9_-]+"),                 # request id
    # Key-anchored salient free-text VALUES. The value is free text, but it is
    # locatable by its key — so a deterministic locker handles it. This is the
    # norm in structured audit logs / traces (Polymorph's domain): the facts an
    # operator needs sit in named fields (msg, root_cause, resolution, error).
    re.compile(
        r'\b(?:root_cause|resolution|resolution_action|remediation|failure_reason'
        r'|reason|msg|message|short_description|summary)\s*[=:]\s*"[^"\n]{1,200}"',
        re.IGNORECASE,
    ),
]


def structural_spans(text: str) -> list[tuple[int, int]]:
    """Merged (start_byte, end_byte) byte ranges of every structural match.

    Offsets are BYTE offsets (utf-8) so they align with ``encode_with_spans``.
    Overlapping/adjacent matches are merged so projection is a simple overlap test.
    """
    raw = text.encode("utf-8")
    ranges: list[tuple[int, int]] = []
    for pat in _LOCK_PATTERNS:
        for m in pat.finditer(text):
            # Convert char offsets to byte offsets (utf-8 is the span alphabet).
            s = len(text[: m.start()].encode("utf-8"))
            e = len(text[: m.end()].encode("utf-8"))
            if e > s:
                ranges.append((s, e))
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def structural_keep_mask(
    text: str, tokenizer: str = "cl100k"
) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    """Return (token_ids, byte_spans, force_keep) for ``text``.

    ``force_keep[i]`` is True iff token i overlaps any structural match — those are
    the tokens the decode floor must never drop. ids/spans come from the same
    ``encode_with_spans`` the labeler and Rust runtime use, so the mask is
    addressable in the identical token space. ``tokenizer`` selects that space
    (cl100k default, or 'modernbert' to match a ModernBERT checkpoint).
    """
    ids, spans = encode_with_spans(text, tokenizer)
    ranges = structural_spans(text)
    force = [False] * len(ids)
    if not ranges:
        return ids, spans, force
    ri = 0
    for i, (a, b) in enumerate(spans):
        # Advance past ranges that end before this token starts.
        while ri < len(ranges) and ranges[ri][1] <= a:
            ri += 1
        if ri < len(ranges) and ranges[ri][0] < b:  # overlap
            force[i] = True
    return ids, spans, force
