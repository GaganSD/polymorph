"""Unit tests for span-aware (chunk-level) decode.

The decisive behaviors:
  * a multi-word phrase with MIXED per-token drop-probs fragments under
    token-level top-k decode but SURVIVES span(min)-decode (the conjunction fix);
  * span decode still hits ~target drop rate on non-needle text;
  * force_keep is honored (a span containing a locked token is never dropped).
"""

from __future__ import annotations

from polymorph_lamr.bench.spandecode import (
    _chunk_spans,
    _word_spans,
    span_decode,
    span_decode_from_text,
)
from polymorph_lamr.bench.methods import _decode_with_drop_order
from polymorph_lamr.label.align import encode_with_spans


def _drop_order_from_probs(probs):
    import numpy as np

    return np.argsort(-np.asarray(probs)).tolist()


def test_word_grouping_basics():
    text = "alpha beta gamma"
    ids, spans = encode_with_spans(text)
    groups = _word_spans(text, spans)
    # Every group is non-empty, all token indices covered exactly once, order kept.
    flat = [i for g in groups for i in g]
    assert flat == list(range(len(ids)))
    # Reconstruct each group's text; should be the 3 whitespace-delimited words
    # (leading spaces fold into the following word per cl100k).
    words = ["".join(text.encode("utf-8")[s:e].decode("utf-8") for s, e in (spans[i] for i in g)).strip() for g in groups]
    assert words == ["alpha", "beta", "gamma"]


def test_multiword_phrase_survives_span_min_but_fragments_token_level():
    # A needle WORD that tokenizes into multiple cl100k subtokens with MIXED
    # drop-probs (one subtoken salient, the rest droppable). Token-level top-k
    # drops the droppable subtokens and shatters the word; span(min) protects the
    # whole word because its MINIMUM subtoken drop-prob is low.
    text = "service failed due to a misconfiguration of the firewall and the proxy layer"
    ids, spans = encode_with_spans(text)
    n = len(ids)

    needle = "misconfiguration"
    nstart = text.index(needle)
    nend = nstart + len(needle)
    needle_tok_idxs = [i for i, (s, e) in enumerate(spans) if s < nend and e > nstart]
    assert len(needle_tok_idxs) >= 2, "needle must split into multiple subtokens"

    # High drop-prob everywhere; only the FIRST subtoken of the needle is salient.
    probs = [0.9] * n
    probs[needle_tok_idxs[0]] = 0.01

    target = 0.5

    tok_out = _decode_with_drop_order(ids, _drop_order_from_probs(probs), target)
    span_out = span_decode(ids, spans, text, probs, target, span="word", aggregator="min")

    # Token-level fragments the word: the high-prob subtoken(s) are dropped, so the
    # contiguous needle string no longer appears.
    assert needle not in tok_out
    # Span(min) keeps it whole: the salient subtoken gives the word a min ~0.01, so
    # the word sits at the bottom of the drop order.
    assert needle in span_out


def test_span_decode_hits_target_rate_on_uniform_text():
    # Long filler with uniform-ish probs: span(min) over words should still drop
    # roughly target fraction of tokens (no needle to protect).
    text = " ".join(f"word{i}" for i in range(200))
    ids, spans = encode_with_spans(text)
    n = len(ids)
    # mild gradient of probs so ordering is well-defined
    probs = [(i % 7) / 7.0 for i in range(n)]
    target = 0.5
    out = span_decode(ids, spans, text, probs, target, span="word", aggregator="min")
    out_ids, _ = encode_with_spans(out)
    achieved = 1.0 - len(out_ids) / n
    # within 15 pts of target (atomic spans + tokenization round-trip drift)
    assert abs(achieved - target) < 0.15


def test_force_keep_span_never_dropped():
    text = "drop me please but keep this token always"
    ids, spans = encode_with_spans(text)
    n = len(ids)
    probs = [0.99] * n  # everything looks maximally droppable
    keep_word = "always"
    kstart = text.index(keep_word)
    kend = kstart + len(keep_word)
    fk = [False] * n
    for i, (s, e) in enumerate(spans):
        if s < kend and e > kstart:
            fk[i] = True
    out = span_decode(ids, spans, text, probs, 0.9, span="word", aggregator="min", force_keep=fk)
    assert keep_word in out


def test_chunk_granularity():
    groups = _chunk_spans(10, 3)
    assert groups == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    text = "alpha beta gamma delta epsilon zeta eta theta"
    ids, spans = encode_with_spans(text)
    probs = [0.5] * len(ids)
    # chunk:2 should run without error and produce a valid decode
    out = span_decode(ids, spans, text, probs, 0.5, span="chunk:2", aggregator="mean")
    assert isinstance(out, str)


def test_aggregator_min_vs_max():
    # A 2-token word: one salient, one droppable. min protects, max drops.
    text = "keepme dropword extra filler tokens here now"
    ids, spans = encode_with_spans(text)
    n = len(ids)
    probs = [0.9] * n
    word = "keepme"
    ws = text.index(word)
    we = ws + len(word)
    word_idxs = [i for i, (s, e) in enumerate(spans) if s < we and e > ws]
    assert len(word_idxs) >= 2  # 'keepme' splits into multiple cl100k tokens
    probs[word_idxs[0]] = 0.0  # one salient token in the word

    min_out = span_decode(ids, spans, text, probs, 0.5, span="word", aggregator="min")
    max_out = span_decode(ids, spans, text, probs, 0.5, span="word", aggregator="max")
    assert word in min_out  # min: protected by the salient token
    assert word not in max_out  # max: the high-prob token makes the word droppable


def test_empty_text():
    assert span_decode_from_text("", [], 0.5) == ""


def test_span_none_unaffected_token_level_baseline():
    # Sanity: span_decode with token granularity reproduces a top-k style drop set
    # equivalent to _decode_with_drop_order (no span grouping benefit).
    text = "one two three four five six seven eight"
    ids, spans = encode_with_spans(text)
    n = len(ids)
    probs = [i / n for i in range(n)]
    tok = _decode_with_drop_order(ids, _drop_order_from_probs(probs), 0.5)
    sp = span_decode(ids, spans, text, probs, 0.5, span="token", aggregator="min")
    assert tok == sp
