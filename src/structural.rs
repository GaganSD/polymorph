//! Deterministic structural locker — the decode-time insurance floor.
//!
//! A small set of regexes over atomic high-salience facts an operator almost
//! always needs in a log: severities, HTTP 4xx/5xx, error/errno codes, IPv4,
//! UUIDs, exception types, incident ids, request ids, and key-anchored salient
//! free-text values. The floor force-keeps every cl100k token overlapping a
//! match so no ranker or budget can drop it. Ported from the Python
//! `polymorph_lamr.bench.structural`.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};

use crate::tokenizer::token_spans;

static LOCK_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let ci = |p: &str| RegexBuilder::new(p).case_insensitive(true).build().unwrap();
    vec![
        Regex::new(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b").unwrap(), // exception type
        Regex::new(r"\b(?:FATAL|CRITICAL|ERROR|EXCEPTION|TRACEBACK|WARN(?:ING)?)\b").unwrap(), // severity
        ci(r"(?:status|HTTP|code)[=:\s]+[45]\d{2}\b"),         // http 4xx/5xx
        Regex::new(r"\berrno[=:]\s*\d+\b").unwrap(),           // errno
        ci(r"\berror code\s+(?:0x[0-9A-Fa-f]+|\d+)\b"),        // error code
        Regex::new(r"\b\d{1,3}(?:\.\d{1,3}){3}\b").unwrap(),   // IPv4
        Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b").unwrap(), // UUID
        Regex::new(r"\bINC\d{4,}\b").unwrap(),                 // incident id
        Regex::new(r"request_id[=:]\s*[A-Za-z0-9_-]+").unwrap(), // request id
        ci(r#"\b(?:root_cause|resolution|resolution_action|remediation|failure_reason|reason|msg|message|short_description|summary)\s*[=:]\s*"[^"\n]{1,200}""#),
    ]
});

/// Merged (start_byte, end_byte) ranges of every structural match. Offsets are
/// UTF-8 byte offsets so they align with [`token_spans`].
pub fn structural_spans(text: &str) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for pat in LOCK_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            // `regex` match offsets are already byte offsets into `text`.
            if m.end() > m.start() {
                ranges.push((m.start(), m.end()));
            }
        }
    }
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort();
    let mut merged = vec![ranges[0]];
    for (s, e) in ranges.into_iter().skip(1) {
        let (ls, le) = *merged.last().unwrap();
        if s <= le {
            *merged.last_mut().unwrap() = (ls, le.max(e));
        } else {
            merged.push((s, e));
        }
    }
    merged
}

/// Returns (token_ids, byte_spans, force_keep) for `text`. `force_keep[i]` is
/// true iff token i overlaps any structural match — the tokens the decode floor
/// must never drop.
pub fn structural_keep_mask(text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>, Vec<bool>)> {
    let (ids, spans) = token_spans(text)?;
    let ranges = structural_spans(text);
    let mut force = vec![false; ids.len()];
    if ranges.is_empty() {
        return Ok((ids, spans, force));
    }
    let mut ri = 0usize;
    for (i, &(a, b)) in spans.iter().enumerate() {
        while ri < ranges.len() && ranges[ri].1 <= a {
            ri += 1;
        }
        if ri < ranges.len() && ranges[ri].0 < b {
            force[i] = true;
        }
    }
    Ok((ids, spans, force))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_merge_overlapping() {
        let text = "ERROR 500 status=503 ok";
        let s = structural_spans(text);
        assert!(!s.is_empty());
        // merged + sorted, non-overlapping
        for w in s.windows(2) {
            assert!(w[0].1 <= w[1].0);
        }
    }

    #[test]
    fn key_anchored_value_is_locked() {
        let text = "INFO note root_cause=\"cascading queue backpressure\" done";
        let (ids, spans, force) = structural_keep_mask(text).unwrap();
        assert_eq!(ids.len(), spans.len());
        assert_eq!(force.len(), ids.len());
        // at least one token inside the quoted value is force-kept
        assert!(force.iter().any(|f| *f));
    }

    #[test]
    fn bare_prose_is_not_locked() {
        let text = "the replication follower silently fell three minutes behind schedule";
        let (_ids, _spans, force) = structural_keep_mask(text).unwrap();
        assert!(force.iter().all(|f| !*f));
    }
}
