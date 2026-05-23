"""Align an original text to its compressed variant, then project the result
back onto cl100k tokens as a binary keep/drop mask.

We use the *token-id* stream from tiktoken as the alignment alphabet — this
keeps the mask cleanly addressable from the downstream Rust runtime which
operates on the same cl100k token IDs (see src/tokens.rs).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def encode_with_spans(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Return cl100k token ids alongside their (start_byte, end_byte) spans
    into `text`. Mirrors src/tokens.rs::token_spans."""
    enc = _enc()
    ids = enc.encode_ordinary(text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    raw = text.encode("utf-8")
    for tok_id in ids:
        b = enc.decode_single_token_bytes(tok_id)
        end = cursor + len(b)
        if raw[cursor:end] != b:
            # Should never happen — cl100k is byte-level BPE; bail loudly.
            raise RuntimeError(
                f"byte-span mismatch at token {tok_id}: cursor={cursor} end={end}"
            )
        spans.append((cursor, end))
        cursor = end
    if cursor != len(raw):
        raise RuntimeError(
            f"byte-span reconstruction mismatch: cursor={cursor} len={len(raw)}"
        )
    return ids, spans


@dataclass(frozen=True)
class AlignmentResult:
    token_ids: list[int]
    spans: list[tuple[int, int]]
    keep_mask: list[bool]   # True = keep (present in compressed), False = drop


def derive_mask(original: str, compressed: str) -> AlignmentResult:
    """LCS-style alignment over cl100k token-id streams. Tokens of `original`
    that match a span in the LCS are marked keep=True; the rest drop=False.

    Note: `keep` here is the *teacher's* binary signal — the trainer will turn
    `keep=False` into the positive class for the "drop" tag through
    label-construction logic.
    """
    orig_ids, spans = encode_with_spans(original)
    comp_ids = _enc().encode_ordinary(compressed)

    keep = [False] * len(orig_ids)
    sm = SequenceMatcher(a=orig_ids, b=comp_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                keep[i] = True

    return AlignmentResult(token_ids=orig_ids, spans=spans, keep_mask=keep)
