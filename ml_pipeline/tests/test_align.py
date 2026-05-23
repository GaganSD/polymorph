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


def test_spans_reconstruct_original():
    text = "Hello, world!\nThis is a test."
    ids, spans = encode_with_spans(text)
    raw = text.encode("utf-8")
    # Spans must be contiguous and cover the whole byte stream.
    assert spans[0][0] == 0
    assert spans[-1][1] == len(raw)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c
