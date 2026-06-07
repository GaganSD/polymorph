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

# Tokenizer backends. The default ("cl100k") keeps the original tiktoken path the
# Rust runtime consumes; "modernbert" emits HF ModernBERT-base token ids so the
# same byte-level alignment can label shards in ModernBERT token space (issue #33).
_MODERNBERT_MODEL = "answerdotai/ModernBERT-base"


# Minimum matched byte-run length to honor when projecting the alignment onto
# tokens. difflib will match coincidental single/near-single bytes (a stray 'e',
# a shared space) in the gaps between real blocks; honoring those would keep dull
# tokens that merely share a letter with the compressed text. Real surviving
# content — and every answer needle we extract (status codes, exception names,
# IPs, ids) — is a contiguous run of at least 3 bytes, so a floor of 3 suppresses
# coincidence without dropping needles.
_MIN_MATCH_RUN = 3

# Whitespace bytes that cl100k attaches to the *following* token (leading space),
# so a matched run's trailing delimiter must not by itself keep the next token.
_WS_BYTES = frozenset(b" \t\r\n")


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=1)
def _modernbert_tok():
    """Lazily load the ModernBERT fast tokenizer. Imported inside the function so
    the default cl100k path never pays the transformers import cost."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_MODERNBERT_MODEL, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError(
            f"{_MODERNBERT_MODEL} did not load a fast tokenizer; offset_mapping "
            "(required for byte-span alignment) is unavailable"
        )
    return tok


def decode_tokens(ids: list[int], tokenizer: str = "cl100k") -> str:
    """Detokenize ``ids`` back to text in the given token space.

    The inverse of ``encode_with_spans``' id stream. ``cl100k`` (default) uses
    tiktoken; ``modernbert`` uses the HF fast tokenizer's ``decode`` (which
    reinserts the byte-level whitespace its sparse offsets omit). Decode-time
    paths (the benchmark, span decode) must detokenize in the SAME space the
    model was trained on, or a ModernBERT checkpoint's surviving ids decode as
    cl100k garbage.
    """
    if tokenizer == "modernbert":
        return _modernbert_tok().decode(ids)
    if tokenizer != "cl100k":
        raise ValueError(f"unknown tokenizer backend: {tokenizer!r}")
    return _enc().decode(ids)


def _char_to_byte_prefix(text: str) -> list[int]:
    """Prefix array mapping a char index -> UTF-8 byte offset of that char's start.
    Length ``len(text)+1``; the last entry is the total byte length. Lets us turn
    HF char offsets into byte offsets exactly, including multi-byte codepoints."""
    prefix = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        prefix[i] = acc
        acc += len(ch.encode("utf-8"))
    prefix[len(text)] = acc
    return prefix


def _encode_with_spans_modernbert(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """ModernBERT token ids + (start_byte, end_byte) spans into `text`.

    Uses the HF fast tokenizer's char-level ``offset_mapping`` and converts char
    offsets to UTF-8 byte offsets. Special tokens (CLS/SEP) carry (0,0) offsets and
    are dropped, so the returned ids/spans cover only real content — matching the
    cl100k contract (every span slices back to its surface bytes). Unlike cl100k,
    ModernBERT spans are NOT guaranteed contiguous / full-cover (it may skip
    whitespace between tokens), so we do not assert reconstruction of the whole
    byte stream; we only assert each token's own bytes round-trip.
    """
    tok = _modernbert_tok()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    raw = text.encode("utf-8")
    c2b = _char_to_byte_prefix(text)
    out_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for tok_id, (cs, ce) in zip(ids, offsets):
        if ce <= cs:
            # Special token or empty span — carries no surface bytes.
            continue
        sb, eb = c2b[cs], c2b[ce]
        # Surface round-trip: the byte span must decode back to the token's text.
        if raw[sb:eb].decode("utf-8") != text[cs:ce]:
            raise RuntimeError(
                f"byte-span mismatch for token {tok_id}: chars=({cs},{ce}) "
                f"bytes=({sb},{eb})"
            )
        out_ids.append(tok_id)
        spans.append((sb, eb))
    return out_ids, spans


def encode_with_spans(
    text: str, tokenizer: str = "cl100k"
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return token ids alongside their (start_byte, end_byte) spans into `text`.

    ``tokenizer="cl100k"`` (default) mirrors src/tokens.rs::token_spans (contiguous,
    full-cover byte-level BPE). ``tokenizer="modernbert"`` emits ModernBERT-base ids
    via the HF fast tokenizer's offset mapping (spans may be sparse over whitespace).
    Both satisfy the same per-token contract: ``text.encode()[start:end]`` is the
    token's surface bytes.
    """
    if tokenizer == "modernbert":
        return _encode_with_spans_modernbert(text)
    if tokenizer != "cl100k":
        raise ValueError(f"unknown tokenizer backend: {tokenizer!r}")
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


def derive_mask(
    original: str, compressed: str, tokenizer: str = "cl100k"
) -> AlignmentResult:
    """Align `original` to `compressed` at the BYTE level, then project the matched
    byte runs onto cl100k tokens: a token is kept if any of its bytes fall in a run
    that survives in `compressed`.

    Why byte-level, not token-id level: an answer needle is a byte substring (an
    exception name, a status code, an IP). The same bytes often tokenize to
    *different* cl100k ids in the original vs the compressed because the
    surrounding bytes differ (BPE merges across boundaries) — and even when the
    ids match, a common sub-token (e.g. ' CR' inside 'CRITICAL') can have its
    keep-budget consumed at another position, dropping the one occurrence that
    forms the needle and breaking the string. Aligning bytes and projecting onto
    token spans sidesteps both: contiguous surviving byte runs keep every token
    that overlaps them, so a needle whose bytes survive in `compressed` survives in
    the label. This is the high-recall objective — a false drop (a lost needle)
    costs far more than a false keep.

    Note: `keep` here is the *teacher's* binary signal — the trainer turns
    `keep=False` into the positive class for the "drop" tag downstream.
    """
    orig_ids, spans = encode_with_spans(original, tokenizer=tokenizer)

    o_bytes = original.encode("utf-8")
    c_bytes = compressed.encode("utf-8")
    matched = bytearray(len(o_bytes))  # 1 where the original byte survives
    # autojunk=False: with only 256 byte values over long inputs, difflib's
    # auto-junk heuristic would treat common bytes (space, newline) as junk and
    # refuse to match them, shredding the alignment. We want every match.
    sm = SequenceMatcher(a=o_bytes, b=c_bytes, autojunk=False)
    for a0, _b0, size in sm.get_matching_blocks():
        if size < _MIN_MATCH_RUN:
            continue  # coincidental short run — don't keep tokens on its account
        for i in range(a0, a0 + size):
            matched[i] = 1

    # Project matched bytes onto tokens. A token is kept if it has a matched
    # NON-whitespace byte, or is matched in full. The non-whitespace rule fixes a
    # boundary artifact: cl100k attaches a leading space to the *next* token, so a
    # matched run ending in a delimiter ("alpha ") would otherwise bleed a single
    # matched space into the following dropped token (" beta") and keep it. The
    # full-match clause still preserves a legitimately-retained whitespace-only
    # token (e.g. indentation inside a kept block).
    keep: list[bool] = []
    for start, end in spans:
        if end <= start:
            keep.append(False)
            continue
        mcount = sum(matched[start:end])
        has_content = any(matched[i] and o_bytes[i] not in _WS_BYTES for i in range(start, end))
        full = mcount == (end - start)
        # Keep if the token is matched in full (preserves whitespace-only tokens
        # inside a kept block) OR a MAJORITY of its bytes survive and at least one
        # is non-whitespace. Majority + non-whitespace rejects single coincidental
        # boundary bytes (a shared trailing letter or a delimiter space bleeding in
        # from an adjacent run) while keeping every token a real surviving run
        # substantially covers.
        keep.append(full or (has_content and 2 * mcount >= (end - start)))
    return AlignmentResult(token_ids=orig_ids, spans=spans, keep_mask=keep)
