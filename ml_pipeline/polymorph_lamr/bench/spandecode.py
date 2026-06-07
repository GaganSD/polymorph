"""Span-aware (chunk-level) decode for the LaMR pruner.

Token-level global top-k decode has a *conjunction failure* on multi-word
needles: it drops a span the moment ANY single token in it crosses the drop
threshold. A free-text root-cause phrase like ``"disk controller firmware
deadlock"`` survives only if every one of its tokens survives, so a single
confidently-droppable filler token ("the", a space) fragments the phrase and the
needle is lost. This is the ChunkKV (ICLR 2025) observation: "existing methods
overlook semantic relationships between tokens resulting in fragmented context."

The fix is to make keep/drop decisions at the granularity of semantic SPANS
(whitespace-delimited words by default, or fixed N-token chunks), dropping whole
spans most-droppable-first until the token budget is hit, never splitting a span.

The governing objective is HIGH RECALL — a false drop of a needle costs far more
than a false keep. The intuitive guess is that ``min`` aggregation (drop a span
only if EVERY token is confidently droppable) maximizes recall by letting one
salient token protect a whole phrase. **It does not.** Measured on the
answer-survival benchmark (R=0.5), the aggregators rank ``max`` 68.5% > ``mean``
62.2% >> ``min`` 23.9%. With a CALIBRATED per-token model, ``min`` is pathological:
it refuses to drop almost every span (most spans contain at least one low-prob
token), so to still hit the target rate it is forced to drop whichever spans
happen to be uniformly mid/high prob — which include real needles — while
keeping confident filler. ``max`` instead trusts the model: it drops the spans
the model flags as droppable and leaves the rest, which preserves more needles at
the same budget. So the default per-span aggregator is ``max`` (matching the Rust
runtime's Word+Max default); ``min`` is retained only as a deliberately
conservative option and for the unit tests that exercise its semantics.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from ..label.align import decode_tokens, encode_with_spans


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


# Whitespace bytes. cl100k attaches a leading space to the *following* token, so
# a word boundary is detected by a non-space->... transition where the token's
# first byte is whitespace (it begins a new word) or the previous token ended in
# a hard whitespace break.
_WS_BYTES = frozenset(b" \t\r\n")


def _word_spans(text: str, spans: list[tuple[int, int]]) -> list[list[int]]:
    """Group token indices into whitespace-delimited words.

    A new word begins whenever a token starts on (or follows) a whitespace
    boundary. cl100k folds a leading space into the next token, so the rule is:
    a token begins a new word if its first byte is whitespace, OR the previous
    token ended in a whitespace byte (a hard newline/tab break), OR it is the
    first token. Whitespace-only tokens attach to the word they precede.
    """
    raw = text.encode("utf-8")
    groups: list[list[int]] = []
    prev_end_ws = True  # before the first token, treat as a boundary
    for i, (s, e) in enumerate(spans):
        if e <= s:
            # empty span — fold into current group if any, else start one
            if groups:
                groups[-1].append(i)
            else:
                groups.append([i])
            continue
        first_ws = raw[s] in _WS_BYTES
        starts_word = (i == 0) or first_ws or prev_end_ws
        if starts_word or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
        prev_end_ws = raw[e - 1] in _WS_BYTES
    return groups


def _chunk_spans(n_tokens: int, chunk_size: int) -> list[list[int]]:
    """Fixed runs of ``chunk_size`` consecutive token indices (the ChunkKV unit)."""
    chunk_size = max(1, chunk_size)
    return [list(range(i, min(i + chunk_size, n_tokens))) for i in range(0, n_tokens, chunk_size)]


def _parse_granularity(span: str, text: str, spans: list[tuple[int, int]], n_tokens: int) -> list[list[int]]:
    if span == "word":
        return _word_spans(text, spans)
    if span.startswith("chunk:"):
        try:
            n = int(span.split(":", 1)[1])
        except ValueError as exc:  # pragma: no cover - guarded by callers
            raise ValueError(f"bad chunk granularity {span!r}") from exc
        return _chunk_spans(n_tokens, n)
    if span == "token":
        return [[i] for i in range(n_tokens)]
    raise ValueError(f"unknown span granularity {span!r} (use 'word' or 'chunk:N')")


_AGGREGATORS = {
    "min": min,   # conservative: drop a span only if EVERY token is droppable
    "mean": lambda xs: sum(xs) / len(xs),
    "max": max,   # default: drop a span if ANY token is droppable — wins on survival
}


def span_decode(
    ids: list[int],
    spans: list[tuple[int, int]],
    text: str,
    drop_probs,
    target_drop_rate: float,
    span: str = "word",
    aggregator: str = "max",
    force_keep: list[bool] | None = None,
    tokenizer: str = "cl100k",
) -> str:
    """Span-aware decode: group tokens into spans, drop whole spans (most-droppable
    first by aggregated drop-prob) until ~``target_drop_rate`` of tokens are gone,
    never splitting a span and never dropping a span that contains a force-kept
    token.

    Args:
      ids:           cl100k token ids.
      spans:         (start_byte, end_byte) per token (from ``encode_with_spans``).
      text:          the original text (for word-boundary detection).
      drop_probs:    per-token P(drop) (any sequence indexable by position).
      target_drop_rate: fraction of TOKENS to remove.
      span:          granularity — 'word' (default) or 'chunk:N' or 'token'.
      aggregator:    per-span score from its token drop-probs — 'max' (default,
                     best survival), 'mean', or 'min' (conservative).
      force_keep:    optional per-token mask; any span containing a True is locked.

    Returns the decoded text of the surviving tokens.
    """
    n = len(ids)
    if n == 0:
        return decode_tokens(ids, tokenizer)
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"unknown aggregator {aggregator!r} (use 'min'/'mean'/'max')")
    agg = _AGGREGATORS[aggregator]
    fk = force_keep or [False] * n

    groups = _parse_granularity(span, text, spans, n)

    # Budget: how many tokens we want to drop.
    k = max(0, round(max(0.0, min(1.0, target_drop_rate)) * n))

    # Score each span; locked spans are never droppable.
    scored: list[tuple[float, int, list[int]]] = []
    for gi, g in enumerate(groups):
        if any(fk[i] for i in g):
            continue  # force-kept span — never dropped
        score = agg([float(drop_probs[i]) for i in g])
        scored.append((score, gi, g))

    # Most-droppable first: highest aggregated drop-prob. Tie-break by span order
    # for determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))

    dropped: set[int] = set()
    for _score, _gi, g in scored:
        if len(dropped) >= k:
            break
        # Atomic: only drop the whole span if it fits in the remaining budget,
        # otherwise skip it and try a smaller span (preserves the rate without
        # splitting a phrase). This keeps achieved-drop close to target while
        # honoring the no-split invariant.
        if len(dropped) + len(g) > k:
            continue
        dropped.update(g)

    return decode_tokens(
        [tid for i, tid in enumerate(ids) if i not in dropped], tokenizer
    )


def span_decode_from_text(
    text: str,
    drop_probs,
    target_drop_rate: float,
    span: str = "word",
    aggregator: str = "max",
    force_keep: list[bool] | None = None,
    tokenizer: str = "cl100k",
) -> str:
    """Convenience wrapper: derive ids/spans from ``text`` via ``encode_with_spans``
    then ``span_decode``. ``drop_probs`` must be aligned to the encoded ids.
    """
    ids, spans = encode_with_spans(text, tokenizer)
    return span_decode(
        ids, spans, text, drop_probs, target_drop_rate,
        span=span, aggregator=aggregator, force_keep=force_keep, tokenizer=tokenizer,
    )
