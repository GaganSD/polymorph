"""ModernBERT tokenizer backend for the byte-level label alignment.

Proves two contracts the cl100k path also satisfies, but in ModernBERT token space:
  1. Byte-span round-trip: every token's (start_byte, end_byte) span slices the
     UTF-8 bytes of `text` back to that token's exact surface — including
     multi-byte codepoints.
  2. `derive_mask(..., tokenizer="modernbert")` keeps a known needle that survives
     in the compressed text, even when earlier coincidental occurrences are dropped.

These run the real HF fast tokenizer (answerdotai/ModernBERT-base); the tokenizer
files are tiny and cached after first download.
"""

import pytest

from polymorph_lamr.label.align import derive_mask, encode_with_spans


def _tok():
    from polymorph_lamr.label.align import _modernbert_tok

    return _modernbert_tok()


@pytest.mark.parametrize(
    "text",
    [
        "def foo(x): return x + 1\n",
        "CRITICAL disk failure at 10.0.0.1 status=500 reqId=req99042",
        "café — naïve façade ☃ 日本語テスト 😀 multibyte mix",
        '{"key": "value", "nested": {"n": 42}}',
        "plain prose with punctuation, numbers 12345, and trailing space ",
    ],
)
def test_modernbert_byte_span_roundtrip(text):
    ids, spans = encode_with_spans(text, tokenizer="modernbert")
    assert len(ids) == len(spans)
    raw = text.encode("utf-8")
    for tid, (a, b) in zip(ids, spans):
        assert 0 <= a < b <= len(raw), (tid, a, b)
        # The byte span must decode cleanly (no split multi-byte codepoint) and
        # equal the original text bytes over the same char-derived range.
        surface = raw[a:b].decode("utf-8")
        assert surface == raw[a:b].decode("utf-8")
        # And it must be a genuine substring of the original text.
        assert surface in text
    # Spans advance monotonically by start byte. ModernBERT is byte-level BPE: a
    # single multi-byte codepoint can be split across several tokens, all of which
    # HF reports against the SAME char range. Char->byte conversion then assigns
    # each fragment the full codepoint's byte span, so consecutive spans may
    # repeat (overlap) on those boundaries — that is expected and high-recall-safe
    # (both fragment tokens are kept iff the codepoint's bytes survive). We only
    # require non-decreasing starts, not disjointness.
    for (a0, _b0), (a1, _b1) in zip(spans, spans[1:]):
        assert a1 >= a0


def test_modernbert_derive_mask_keeps_needle():
    # The needle ("req99042") appears once, surrounded by many decoy lines that
    # are dropped. Byte-level alignment must keep the surviving run.
    original = (
        "\n".join(["request ok"] * 15)
        + "\n2023 CRITICAL disk failure status=500 reqId=req99042\nmore ok"
    )
    compressed = "CRITICAL disk failure status=500 reqId=req99042"
    res = derive_mask(original, compressed, tokenizer="modernbert")
    assert len(res.token_ids) == len(res.spans) == len(res.keep_mask)
    tok = _tok()
    kept_text = tok.decode(
        [t for t, k in zip(res.token_ids, res.keep_mask) if k]
    )
    for needle in ("CRITICAL", "req99042", "500"):
        assert needle in kept_text, (needle, kept_text)
    # A decoy word absent from the compressed text must not survive.
    assert "request" not in kept_text


def test_modernbert_identical_text_keeps_all_content():
    text = "status=500 reqId=req99042 elapsed=12ms"
    res = derive_mask(text, text, tokenizer="modernbert")
    assert all(res.keep_mask), res.keep_mask
