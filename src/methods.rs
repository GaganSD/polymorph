//! Compression methods compared by the answer-survival benchmark.
//!
//! Each method takes a log chunk and a target drop rate R (fraction of tokens to
//! remove) and returns the compressed text. Survival is then measured on that
//! text (see `survival.rs`). Always-available, GPU-free, deterministic:
//!   * [`DeterministicDedup`]    — line-normalize + run-length collapse.
//!   * [`KeepSeverityHeuristic`] — keep the most-severe lines until the budget.
//!   * [`RandomDropFloor`]       — deterministic pseudo-random token drop
//!     (the floor any real ranker must beat).
//!
//! The optional torch-backed `LaMRMethod` and network `LLMLingua2Method` from the
//! Python original are intentionally out of scope for the Rust runtime. Ported
//! from `polymorph_lamr.bench.methods`.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::spandecode::{span_decode, Aggregator};
use crate::structural::structural_keep_mask;
use crate::tokenizer::{count_tokens, decode_tokens, token_spans};

type TokenizedWithFloor = (Vec<u32>, Vec<(usize, usize)>, Option<Vec<bool>>);

/// Count cl100k tokens in `text`.
pub fn token_count(text: &str) -> usize {
    count_tokens(text).unwrap_or(0)
}

/// A compression method: encode/transform `text` to hit `target_drop_rate`.
pub trait CompressionMethod {
    fn name(&self) -> &str;
    fn tunable(&self) -> bool;
    /// `(is_available, reason_if_not)`.
    fn available(&self) -> (bool, String) {
        (true, String::new())
    }
    fn compress(&self, text: &str, target_drop_rate: f64) -> Result<String>;
}

/// Python's `round` is round-half-to-even; reproduce it for budget/k parity.
fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.floor() - 0.5).abs() < f64::EPSILON {
        // exactly .5 — round to even
        let f = x.floor() as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    } else {
        r as i64
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3, reflected) — matches Python's `zlib.crc32`.
// ---------------------------------------------------------------------------

static CRC_TABLE: Lazy<[u32; 256]> = Lazy::new(|| {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
});

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = CRC_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Deterministic dedup (mirror of src/dedup.rs's normalization classes)
// ---------------------------------------------------------------------------

static NORM_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        (
            Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
                .unwrap(),
            "<TS>",
        ),
        (
            Regex::new(r"\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]\d{4}").unwrap(),
            "<TS>",
        ),
        (
            Regex::new(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            )
            .unwrap(),
            "<UUID>",
        ),
        (Regex::new(r"\b\d{1,3}(?:\.\d{1,3}){3}\b").unwrap(), "<IP>"),
        (Regex::new(r"\b0x[0-9a-fA-F]+\b").unwrap(), "<HEX>"),
        (Regex::new(r"\b\d+\b").unwrap(), "<NUM>"),
    ]
});

fn normalize_key(line: &str) -> String {
    let mut key = line.to_string();
    for (pat, repl) in NORM_PATTERNS.iter() {
        key = pat.replace_all(&key, *repl).into_owned();
    }
    key.trim().to_string()
}

/// Collapse runs of consecutive lines sharing a normalized template into
/// head + "... N lines elided ..." + tail.
pub struct DeterministicDedup {
    pub min_run: usize,
}

impl Default for DeterministicDedup {
    fn default() -> Self {
        Self { min_run: 3 }
    }
}

impl CompressionMethod for DeterministicDedup {
    fn name(&self) -> &str {
        "deterministic"
    }
    fn tunable(&self) -> bool {
        false
    }
    fn compress(&self, text: &str, _target_drop_rate: f64) -> Result<String> {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Ok(text.to_string());
        }
        let mut out: Vec<String> = Vec::new();
        let n = lines.len();
        let mut i = 0;
        while i < n {
            let key = normalize_key(lines[i]);
            let mut j = i + 1;
            while j < n && normalize_key(lines[j]) == key {
                j += 1;
            }
            let run = j - i;
            if run >= self.min_run {
                out.push(lines[i].to_string());
                out.push(format!("... {} lines elided ...", run - 2));
                out.push(lines[j - 1].to_string());
            } else {
                for ln in &lines[i..j] {
                    out.push(ln.to_string());
                }
            }
            i = j;
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Keep-severity heuristic
// ---------------------------------------------------------------------------

const SEVERITY_ORDER: &[&str] = &[
    "FATAL",
    "CRITICAL",
    "ERROR",
    "EXCEPTION",
    "TRACEBACK",
    "WARN",
];

fn severity_rank(line: &str) -> usize {
    let up = line.to_uppercase();
    for (rank, kw) in SEVERITY_ORDER.iter().enumerate() {
        if up.contains(kw) {
            return rank;
        }
    }
    SEVERITY_ORDER.len()
}

/// Keep the most-severe lines first until the keep budget is spent; emit kept
/// lines in original order.
#[derive(Default)]
pub struct KeepSeverityHeuristic;

impl CompressionMethod for KeepSeverityHeuristic {
    fn name(&self) -> &str {
        "keep-severity"
    }
    fn tunable(&self) -> bool {
        true
    }
    fn compress(&self, text: &str, target_drop_rate: f64) -> Result<String> {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Ok(text.to_string());
        }
        let total = token_count(text);
        let budget = round_half_even((1.0 - target_drop_rate) * total as f64).max(1);
        // Stable priority: (severity_rank, original_index).
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by_key(|&idx| (severity_rank(lines[idx]), idx));
        let mut keep: Vec<bool> = vec![false; lines.len()];
        let mut spent: i64 = 0;
        for idx in order {
            let cost = token_count(lines[idx]) as i64 + 1;
            if spent + cost > budget && keep.iter().any(|k| *k) {
                continue;
            }
            keep[idx] = true;
            spent += cost;
            if spent >= budget {
                break;
            }
        }
        let kept: Vec<&str> = lines
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, l)| *l)
            .collect();
        Ok(kept.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Random-drop floor (deterministic)
// ---------------------------------------------------------------------------

/// Drop the first `round(R*n)` tokens in `drop_order` (most-droppable first),
/// skipping any token marked in `force_keep`, then decode the survivors.
fn decode_with_drop_order(
    ids: &[u32],
    drop_order: &[usize],
    target_drop_rate: f64,
    force_keep: Option<&[bool]>,
) -> Result<String> {
    let n = ids.len();
    if n == 0 {
        return decode_tokens(ids);
    }
    let rate = target_drop_rate.clamp(0.0, 1.0);
    let k = (round_half_even(rate * n as f64).max(0) as usize).min(n);
    let fk_owned;
    let fk: &[bool] = match force_keep {
        Some(f) => f,
        None => {
            fk_owned = vec![false; n];
            &fk_owned
        }
    };
    let mut dropped: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &i in drop_order {
        if dropped.len() >= k {
            break;
        }
        if fk[i] {
            continue;
        }
        dropped.insert(i);
    }
    let survivors: Vec<u32> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| !dropped.contains(i))
        .map(|(_, &tid)| tid)
        .collect();
    decode_tokens(&survivors)
}

/// Deterministically drop ~R of the tokens (keyed by content + index). With
/// `floor=true` the structural locker force-keeps high-salience atoms.
pub struct RandomDropFloor {
    name: String,
    pub floor: bool,
    pub span: Option<String>,
    pub aggregator: Aggregator,
}

impl RandomDropFloor {
    pub fn new(floor: bool, span: Option<String>, aggregator: Aggregator) -> Self {
        let mut name = String::from("random");
        if span.is_some() {
            name.push_str("+span");
        }
        if floor {
            name.push_str("+floor");
        }
        Self {
            name,
            floor,
            span,
            aggregator,
        }
    }
}

impl Default for RandomDropFloor {
    fn default() -> Self {
        Self::new(false, None, Aggregator::Max)
    }
}

impl CompressionMethod for RandomDropFloor {
    fn name(&self) -> &str {
        &self.name
    }
    fn tunable(&self) -> bool {
        true
    }
    fn compress(&self, text: &str, target_drop_rate: f64) -> Result<String> {
        let (ids, spans, force_keep): TokenizedWithFloor = if self.floor {
            let (ids, spans, fk) = structural_keep_mask(text)?;
            (ids, spans, Some(fk))
        } else if self.span.is_some() {
            let (ids, spans) = token_spans(text)?;
            (ids, spans, None)
        } else {
            let (ids, _spans) = token_spans(text)?;
            (ids, Vec::new(), None)
        };
        if ids.is_empty() {
            return Ok(text.to_string());
        }
        let seed = crc32(text.as_bytes());
        let crc: Vec<u32> = (0..ids.len())
            .map(|i| crc32(format!("{seed}:{i}").as_bytes()))
            .collect();

        if let Some(span) = &self.span {
            // Higher drop_prob = more droppable. Lower crc was more droppable, so
            // invert: drop_prob = 1 - crc/2^32.
            let drop_probs: Vec<f64> = crc
                .iter()
                .map(|&c| 1.0 - (c as f64 / 4_294_967_296.0))
                .collect();
            return span_decode(
                &ids,
                &spans,
                text,
                &drop_probs,
                target_drop_rate,
                span,
                self.aggregator,
                force_keep.as_deref(),
            );
        }
        // Most-droppable first: ascending crc.
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| crc[i]);
        decode_with_drop_order(&ids, &order, target_drop_rate, force_keep.as_deref())
    }
}

/// The standard comparison set of always-available baselines.
pub fn default_methods() -> Vec<Box<dyn CompressionMethod>> {
    vec![
        Box::new(DeterministicDedup::default()),
        Box::new(KeepSeverityHeuristic),
        Box::new(RandomDropFloor::default()),
        Box::new(RandomDropFloor::new(true, None, Aggregator::Max)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survival::achieved_drop_rate;

    #[test]
    fn crc32_matches_zlib_known_vectors() {
        // zlib.crc32(b"") == 0; zlib.crc32(b"123456789") == 0xCBF43926.
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"hello"), 0x3610_A686);
    }

    #[test]
    fn deterministic_dedup_collapses_runs_and_keeps_uniques() {
        let mut lines: Vec<String> = (0..20)
            .map(|i| format!("2023-01-01T00:00:0{}.0 INFO heartbeat ok", i % 10))
            .collect();
        lines.push("FATAL unique crash code 0xDEAD".to_string());
        let text = lines.join("\n");
        let out = DeterministicDedup::default().compress(&text, 0.0).unwrap();
        assert!(token_count(&out) < token_count(&text));
        assert!(out.contains("0xDEAD"));
        assert!(out.contains("elided"));
    }

    #[test]
    fn keep_severity_keeps_severe_line_and_shrinks() {
        let mut lines: Vec<String> = (0..40).map(|i| format!("GET /x 200 ok id={i}")).collect();
        lines.push("ERROR 500 failure token=NEEDLE42".to_string());
        let text = lines.join("\n");
        let out = KeepSeverityHeuristic.compress(&text, 0.7).unwrap();
        assert!(token_count(&out) < token_count(&text));
        assert!(out.contains("NEEDLE42"));
    }

    #[test]
    fn random_drop_is_deterministic_and_hits_rate() {
        let text = (0..400)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let a = RandomDropFloor::default().compress(&text, 0.5).unwrap();
        let b = RandomDropFloor::default().compress(&text, 0.5).unwrap();
        assert_eq!(a, b);
        let drop = achieved_drop_rate(&text, &a);
        assert!(0.4 < drop && drop < 0.6, "drop={drop}");
    }

    #[test]
    fn random_drop_rate_zero_keeps_everything() {
        let text = (0..100)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let out = RandomDropFloor::default().compress(&text, 0.0).unwrap();
        assert!(achieved_drop_rate(&text, &out) < 0.05);
    }

    #[test]
    fn floor_locks_key_anchored_value_but_not_unanchored_prose() {
        let floored = RandomDropFloor::new(true, None, Aggregator::Max);
        let mut keyed = (0..40)
            .map(|i| format!("INFO tick {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        keyed.push_str("\nINFO note root_cause=\"cascading queue backpressure\" done");
        let out = floored.compress(&keyed, 0.8).unwrap();
        assert!(crate::survival::answer_survives(
            "cascading queue backpressure",
            &out
        ));

        let mut bare = (0..40)
            .map(|i| format!("INFO tick {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        bare.push_str("\nthe replication follower silently fell three minutes behind schedule");
        let out2 = floored.compress(&bare, 0.8).unwrap();
        assert!(!crate::survival::answer_survives(
            "silently fell three minutes behind",
            &out2
        ));
    }
}
