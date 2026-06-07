"""Alignment round-trips a known mask: a strict subsequence of cl100k tokens
must be marked keep=True; deleted tokens must be keep=False."""

from polymorph_lamr.label.align import derive_mask, encode_with_spans


def test_identical_strings_keep_everything():
    text = "def foo(x): return x + 1\n"
    res = derive_mask(text, text)
    assert all(res.keep_mask), f"expected all keep, got {res.keep_mask}"
    assert len(res.token_ids) == len(res.spans) == len(res.keep_mask)


def test_dropping_middle_word_marks_correct_tokens():
    original = "alpha beta gamma delta epsilon"
    compressed = "alpha gamma epsilon"
    res = derive_mask(original, compressed)
    # Reconstruct kept substring from token ids.
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    kept_text = enc.decode(
        [tok for tok, keep in zip(res.token_ids, res.keep_mask) if keep]
    )
    # cl100k tokenises with leading spaces; the kept stream must be a substring-
    # equivalent to "alpha gamma epsilon" modulo leading whitespace.
    assert "alpha" in kept_text
    assert "gamma" in kept_text
    assert "epsilon" in kept_text
    # 'beta' and 'delta' should not appear at all.
    assert "beta" not in kept_text
    assert "delta" not in kept_text


def _kept_text(original: str, compressed: str) -> str:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    res = derive_mask(original, compressed)
    return enc.decode([t for t, k in zip(res.token_ids, res.keep_mask) if k])


def test_needle_survives_when_subtoken_budget_consumed_elsewhere():
    # "CRITICAL" tokenizes as [' CR', 'ITICAL']; the common ' CR' sub-token also
    # appears in many other words. Token-id alignment would spend ' CR' elsewhere
    # and drop the one forming the needle, breaking the string. Byte-level
    # alignment keeps the whole surviving run.
    original = (
        "CREATE ok\nCRAWL ok\nCRC ok\n2023 CRITICAL disk failure\nCROP ok\nCRUST ok"
    )
    compressed = "CRITICAL disk failure"
    assert "CRITICAL" in _kept_text(original, compressed)


def test_repeated_token_needle_survives():
    # The kept occurrence of a repeated word must survive even when earlier
    # occurrences are dropped.
    original = "\n".join(["request ok"] * 20 + ["request FAILED id=req99042"])
    compressed = "request FAILED id=req99042"
    assert "req99042" in _kept_text(original, compressed)


def test_word_absent_from_compressed_is_dropped():
    # A word that does not appear in the compressed text at all stays dropped
    # (majority-coverage rejects coincidental single boundary bytes).
    kept = _kept_text("send payload now please", "send now")
    assert "payload" not in kept
    assert "please" not in kept


def test_spans_reconstruct_original():
    text = "Hello, world!\nThis is a test."
    ids, spans = encode_with_spans(text)
    raw = text.encode("utf-8")
    # Spans must be contiguous and cover the whole byte stream.
    assert spans[0][0] == 0
    assert spans[-1][1] == len(raw)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c
