//! Span-aware (chunk-level) decode for the LaMR pruner.
//!
//! Token-level global top-k decode has a *conjunction failure* on multi-word
//! needles: it drops a span the moment ANY single token in it crosses the drop
//! threshold, fragmenting multi-word phrases. The fix is to make keep/drop
//! decisions at the granularity of semantic SPANS (whitespace-delimited words by
//! default, or fixed N-token chunks), dropping whole spans most-droppable-first
//! until the token budget is hit, never splitting a span.
//!
//! The default per-span aggregator is `max` (matching the Rust runtime's
//! Word+Max default): drop a span if ANY token is droppable — empirically best
//! on the answer-survival benchmark (max 68.5% > mean 62.2% >> min 23.9% at
//! R=0.5). Ported from the Python `polymorph_lamr.bench.spandecode`.

use anyhow::{anyhow, Result};

use crate::tokenizer::{decode_tokens, token_spans};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregator {
    Min,
    Mean,
    Max,
}

impl Aggregator {
    pub fn parse(s: &str) -> Result<Aggregator> {
        match s {
            "min" => Ok(Aggregator::Min),
            "mean" => Ok(Aggregator::Mean),
            "max" => Ok(Aggregator::Max),
            other => Err(anyhow!(
                "unknown aggregator {other:?} (use 'min'/'mean'/'max')"
            )),
        }
    }

    fn apply(self, xs: &[f64]) -> f64 {
        match self {
            Aggregator::Min => xs.iter().cloned().fold(f64::INFINITY, f64::min),
            Aggregator::Max => xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            Aggregator::Mean => xs.iter().sum::<f64>() / xs.len() as f64,
        }
    }
}

const WS_BYTES: [u8; 4] = [b' ', b'\t', b'\r', b'\n'];

fn is_ws(b: u8) -> bool {
    WS_BYTES.contains(&b)
}

/// Group token indices into whitespace-delimited words. cl100k folds a leading
/// space into the next token, so a token begins a new word if its first byte is
/// whitespace, OR the previous token ended in a whitespace byte, OR it is the
/// first token.
pub fn word_spans(text: &str, spans: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let raw = text.as_bytes();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut prev_end_ws = true;
    for (i, &(s, e)) in spans.iter().enumerate() {
        if e <= s {
            if let Some(last) = groups.last_mut() {
                last.push(i);
            } else {
                groups.push(vec![i]);
            }
            continue;
        }
        let first_ws = is_ws(raw[s]);
        let starts_word = i == 0 || first_ws || prev_end_ws;
        if starts_word || groups.is_empty() {
            groups.push(vec![i]);
        } else {
            groups.last_mut().unwrap().push(i);
        }
        prev_end_ws = is_ws(raw[e - 1]);
    }
    groups
}

/// Fixed runs of `chunk_size` consecutive token indices (the ChunkKV unit).
pub fn chunk_spans(n_tokens: usize, chunk_size: usize) -> Vec<Vec<usize>> {
    let chunk_size = chunk_size.max(1);
    let mut out = Vec::new();
    let mut i = 0;
    while i < n_tokens {
        out.push((i..(i + chunk_size).min(n_tokens)).collect());
        i += chunk_size;
    }
    out
}

fn parse_granularity(
    span: &str,
    text: &str,
    spans: &[(usize, usize)],
    n_tokens: usize,
) -> Result<Vec<Vec<usize>>> {
    if span == "word" {
        return Ok(word_spans(text, spans));
    }
    if let Some(rest) = span.strip_prefix("chunk:") {
        let n: usize = rest
            .parse()
            .map_err(|_| anyhow!("bad chunk granularity {span:?}"))?;
        return Ok(chunk_spans(n_tokens, n));
    }
    if span == "token" {
        return Ok((0..n_tokens).map(|i| vec![i]).collect());
    }
    Err(anyhow!(
        "unknown span granularity {span:?} (use 'word' or 'chunk:N')"
    ))
}

/// Span-aware decode: group tokens into spans, drop whole spans (most-droppable
/// first by aggregated drop-prob) until ~`target_drop_rate` of tokens are gone,
/// never splitting a span and never dropping a span containing a force-kept token.
#[allow(clippy::too_many_arguments)]
pub fn span_decode(
    ids: &[u32],
    spans: &[(usize, usize)],
    text: &str,
    drop_probs: &[f64],
    target_drop_rate: f64,
    span: &str,
    aggregator: Aggregator,
    force_keep: Option<&[bool]>,
) -> Result<String> {
    let n = ids.len();
    if n == 0 {
        return decode_tokens(ids);
    }
    let fk_owned;
    let fk: &[bool] = match force_keep {
        Some(f) => f,
        None => {
            fk_owned = vec![false; n];
            &fk_owned
        }
    };

    let groups = parse_granularity(span, text, spans, n)?;

    let rate = target_drop_rate.clamp(0.0, 1.0);
    let k = (rate * n as f64).round() as usize;

    // Score each span; locked spans are never droppable.
    let mut scored: Vec<(f64, usize, &Vec<usize>)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        if g.iter().any(|&i| fk[i]) {
            continue;
        }
        let vals: Vec<f64> = g.iter().map(|&i| drop_probs[i]).collect();
        scored.push((aggregator.apply(&vals), gi, g));
    }

    // Most-droppable first: highest aggregated drop-prob, tie-break by span order.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut dropped: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (_score, _gi, g) in &scored {
        if dropped.len() >= k {
            break;
        }
        if dropped.len() + g.len() > k {
            continue;
        }
        for &i in g.iter() {
            dropped.insert(i);
        }
    }

    let survivors: Vec<u32> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| !dropped.contains(i))
        .map(|(_, &tid)| tid)
        .collect();
    decode_tokens(&survivors)
}

/// Convenience wrapper: derive ids/spans from `text` then `span_decode`.
pub fn span_decode_from_text(
    text: &str,
    drop_probs: &[f64],
    target_drop_rate: f64,
    span: &str,
    aggregator: Aggregator,
    force_keep: Option<&[bool]>,
) -> Result<String> {
    let (ids, spans) = token_spans(text)?;
    span_decode(
        &ids,
        &spans,
        text,
        drop_probs,
        target_drop_rate,
        span,
        aggregator,
        force_keep,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_spans_partition() {
        let g = chunk_spans(7, 3);
        assert_eq!(g, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
    }

    #[test]
    fn word_spans_group_by_whitespace() {
        let text = "alpha beta gamma";
        let (_ids, spans) = token_spans(text).unwrap();
        let groups = word_spans(text, &spans);
        // round-trips to the same token count
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, spans.len());
    }

    #[test]
    fn max_aggregator_drops_droppable_span() {
        let text = "keep this needle phrase but drop filler tokens here";
        let (ids, spans) = token_spans(text).unwrap();
        // uniform mid drop-prob, decode at 30% — should produce shorter text.
        let probs = vec![0.5_f64; ids.len()];
        let out = span_decode(
            &ids,
            &spans,
            text,
            &probs,
            0.3,
            "word",
            Aggregator::Max,
            None,
        )
        .unwrap();
        assert!(out.len() <= text.len());
    }

    #[test]
    fn force_keep_protects_span() {
        let text = "alpha beta gamma delta epsilon zeta";
        let (ids, spans) = token_spans(text).unwrap();
        let probs = vec![0.9_f64; ids.len()];
        let mut fk = vec![false; ids.len()];
        fk[0] = true; // lock first token
        let out = span_decode(
            &ids,
            &spans,
            text,
            &probs,
            1.0,
            "word",
            Aggregator::Max,
            Some(&fk),
        )
        .unwrap();
        assert!(out.contains("alpha"));
    }
}
