//! Training-time normalization + trash detection for the distillation sampler.
//!
//! Two jobs: **template keying** (`normalize_line` / `template_key`) collapses
//! lines that are structurally identical modulo their variable tokens so a corpus
//! of near-duplicate records dedups to its handful of real templates; and **trash
//! gating** (`signal_ratio` / `is_low_signal`) drops lines that carry no real
//! linguistic signal (synthetic random-blob payloads). Ported from the Python
//! `polymorph_lamr.distill.normalize`; `normalize_line` mirrors the runtime
//! `src/dedup.rs` masking classes.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

// Rust-mirror normalization patterns (order matters; see src/dedup.rs).
static NORMALIZE_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
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
        (Regex::new(r"\b0[xX][0-9a-fA-F]+\b").unwrap(), "<HEX>"),
        (Regex::new(r"\b[0-9a-fA-F]{16,}\b").unwrap(), "<HEX>"),
        (Regex::new(r"\b\d+(?:\.\d+)?\b").unwrap(), "<NUM>"),
    ]
});

const PLACEHOLDERS: &[&str] = &["TS", "UUID", "IP", "HEX", "NUM", "RAND"];

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'A' | 'E' | 'I' | 'O' | 'U' | 'Y'
    )
}

static ALNUM_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9]+").unwrap());
static ANY_DIGIT_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+").unwrap());
static PLACEHOLDER_RUN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<(?:TS|UUID|IP|HEX|NUM|RAND)>(?:\s+<(?:TS|UUID|IP|HEX|NUM|RAND)>)+").unwrap()
});

/// Mask variable tokens into a normalized template key. Faithful mirror of
/// `src/dedup.rs::normalize_line`: fixed pattern set applied in fixed order.
pub fn normalize_line(line: &str) -> String {
    let mut out = line.to_string();
    for (pat, repl) in NORMALIZE_PATTERNS.iter() {
        out = pat.replace_all(&out, *repl).into_owned();
    }
    out
}

fn entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let n = s.chars().count() as f64;
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    -counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

fn case_transition_rate(s: &str) -> f64 {
    let letters: Vec<char> = s.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() < 2 {
        return 0.0;
    }
    let flips = letters
        .windows(2)
        .filter(|w| w[0].is_lowercase() != w[1].is_lowercase())
        .count();
    flips as f64 / (letters.len() - 1) as f64
}

fn max_consonant_run(s: &str) -> usize {
    let mut best = 0;
    let mut run = 0;
    for c in s.chars() {
        if c.is_alphabetic() && !is_vowel(c) {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

/// True if `tok` looks like a high-entropy random blob (no linguistic signal).
/// A vote over independent randomness signals; two or more ⇒ random. Tokens
/// under 12 chars are never confidently random.
pub fn is_random_token(tok: &str) -> bool {
    if tok.chars().count() < 12 {
        return false;
    }
    let letters: Vec<char> = tok.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }
    let vowel_count = letters.iter().filter(|c| is_vowel(**c)).count();
    let vowel_ratio = vowel_count as f64 / letters.len() as f64;
    let has_digit = tok.chars().any(|c| c.is_ascii_digit());
    let signals = [
        vowel_ratio < 0.30,
        case_transition_rate(tok) >= 0.30,
        max_consonant_run(tok) >= 5,
        entropy(tok) >= 4.0,
        has_digit && letters.len() >= 4,
    ];
    signals.iter().filter(|s| **s).count() >= 2
}

/// Replace random-blob alphanumeric runs with `<RAND>`. Runs after
/// `normalize_line`.
pub fn mask_random(text: &str) -> String {
    ALNUM_RUN
        .replace_all(text, |caps: &regex::Captures| {
            let run = &caps[0];
            if is_random_token(run) {
                "<RAND>".to_string()
            } else {
                run.to_string()
            }
        })
        .into_owned()
}

/// Sampling-time dedup key: normalize variables, then mask random blobs, then
/// fold underscore-glued digit runs and collapse adjacent placeholder runs.
pub fn template_key(line: &str) -> String {
    let masked = mask_random(&normalize_line(line));
    let key = ANY_DIGIT_RUN.replace_all(&masked, "<NUM>").into_owned();
    PLACEHOLDER_RUN.replace_all(&key, "<RAND>").into_owned()
}

/// `template_key` (the Python LRU is a perf detail; behavior is identical).
pub fn template_key_cached(line: &str) -> String {
    template_key(line)
}

fn is_wordlike(tok: &str) -> bool {
    if PLACEHOLDERS.contains(&tok) {
        return true;
    }
    let letters: Vec<char> = tok.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }
    if is_random_token(tok) {
        return false;
    }
    let vowel_count = letters.iter().filter(|c| is_vowel(**c)).count();
    let vowel_ratio = vowel_count as f64 / letters.len() as f64;
    if tok.chars().count() >= 12 && vowel_ratio < 0.20 {
        return false;
    }
    true
}

/// Fraction of alphanumeric characters that belong to word-like tokens. Returns
/// 1.0 for text with no alphanumeric content.
pub fn signal_ratio(text: &str) -> f64 {
    let runs: Vec<&str> = ALNUM_RUN.find_iter(text).map(|m| m.as_str()).collect();
    let total: usize = runs.iter().map(|r| r.chars().count()).sum();
    if total == 0 {
        return 1.0;
    }
    let signal: usize = runs
        .iter()
        .filter(|r| is_wordlike(r))
        .map(|r| r.chars().count())
        .sum();
    signal as f64 / total as f64
}

/// True if `text` is trash: enough content to judge, but no real signal.
pub fn is_low_signal(text: &str, min_ratio: f64, min_alnum: usize) -> bool {
    let total: usize = ALNUM_RUN
        .find_iter(text)
        .map(|m| m.as_str().chars().count())
        .sum();
    if total < min_alnum {
        return false;
    }
    signal_ratio(text) < min_ratio
}

/// `is_low_signal` with the default thresholds (min_ratio=0.30, min_alnum=24).
pub fn is_low_signal_default(text: &str) -> bool {
    is_low_signal(text, 0.30, 24)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANDOM_BLOBS: &[&str] = &[
        "RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z",
        "2eykZAqkscUJ4KUxihlYXskJiDSG3TF0CXhFXZp9TfdtWDQWvfeJjXS1wiOikTyhDKEPIUsTbXRLcN9",
        "5r2039obpbhM8gNiE2GPONpWhYnkQBqFoM0IhXlPIqSkoJEnFqibMcbW8yEetJWD3tKbM4RvNixxpustB2d6p2HvFYhQ9fN",
        "hqMA49ZWCXsrHsXy81t5pCoPJBer4QpdnJdelM7WPCykUZENL6T469HHsAybKVxlk1JyuWWQLf3x3eRPFkiO8LKPIyMEfFtC4iKqVTzy9nAKMz0ziBAIFCXD",
        "xuqbiprmtjwu",
    ];

    const REAL_VALUES: &[&str] = &[
        "Database",
        "connection",
        "failure",
        "Security",
        "Scan",
        "Failure",
        "Jenkins",
        "inventory",
        "Internal",
        "server",
        "error",
        "Unauthorized",
        "Optimize",
        "Resource",
        "Exhaustion",
        "windows",
        "latest",
        "Python",
        "getUserAccountBalance",
        "synchronized",
        "authentication",
        "getProfileByUserName",
        "team",
        "alpha",
        "pipeline",
    ];

    #[test]
    fn random_blobs_are_flagged() {
        for blob in RANDOM_BLOBS {
            assert!(is_random_token(blob), "expected {blob:?} flagged as random");
        }
    }

    #[test]
    fn real_values_are_not_flagged() {
        for word in REAL_VALUES {
            assert!(!is_random_token(word), "{word:?} wrongly flagged as random");
        }
    }

    #[test]
    fn short_tokens_never_flagged() {
        for tok in ["lljugd", "mckssy", "ERR_621", "repo", "200", "OK", "ab"] {
            assert!(!is_random_token(tok));
        }
    }

    #[test]
    fn normalize_line_mirrors_rust_patterns() {
        let line = r#"233.223.117.90 - - [27/Dec/2037:12:00:00 +0530] "GET /x" 200 42"#;
        let key = normalize_line(line);
        assert!(key.contains("<IP>") && key.contains("<TS>") && key.contains("<NUM>"));
        let other = r#"162.253.4.179 - - [27/Dec/2037:13:00:00 +0530] "GET /x" 200 99"#;
        assert_eq!(normalize_line(line), normalize_line(other));
    }

    #[test]
    fn normalize_masks_iso_ts_uuid_hex() {
        let line = "2025-12-29T07:58:16.927259 commit \
            53820d0dddb2d97f40cbf0e1b4566169f480b86e id 0xDEADBEEF \
            550e8400-e29b-41d4-a716-446655440000";
        let key = normalize_line(line);
        assert!(key.contains("<TS>"));
        assert!(key.contains("<HEX>"));
        assert!(key.contains("<UUID>"));
        assert!(!key.contains("53820d0"));
    }

    #[test]
    fn template_key_collapses_cicd_rows_modulo_blob() {
        let a = "2025-12-29T07:58:16 MEDIUM Jenkins stage=deploy \
            error_code=ERR_621 msg=\"ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z\"";
        let b = "2026-01-02T05:36:16 MEDIUM Jenkins stage=deploy \
            error_code=ERR_621 msg=\"ERROR: g8tprJBHkhLu3r6EtkU5E0Y51Gda8lfG5iCOMWoFH\"";
        assert_eq!(template_key(a), template_key(b));
        assert!(template_key(a).contains("<RAND>"));
    }

    #[test]
    fn template_key_collapses_variable_segment_count() {
        let one = "stage=deploy msg=\"ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z\"";
        let two = "stage=deploy msg=\"ERROR: g8tprJBHkhLu3r6EtkU5E0Y51Gda8lfG5iCOMWoFH \
            2eykZAqkscUJ4KUxihlYXskJiDSG3TF0CXhFXZp9Tf\"";
        let three = "stage=deploy msg=\"ERROR: XSxiDN8XrGhYd9legdlv1fCtI9ILHaTM94tucUUNwL \
            dfutS465iBuUtmcDowLmZts0LG4y70lpe4I6iefsVT \
            RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z\"";
        assert_eq!(template_key(one), template_key(two));
        assert_eq!(template_key(two), template_key(three));
    }

    #[test]
    fn template_key_keeps_distinct_templates_distinct() {
        let a = "MEDIUM Jenkins stage=deploy failure_type=Security Scan Failure";
        let b = "CRITICAL GitLab stage=build failure_type=Network Error";
        assert_ne!(template_key(a), template_key(b));
    }

    #[test]
    fn mask_random_preserves_structure() {
        let line = "msg=\"ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z\" stage=deploy";
        let masked = mask_random(line);
        assert!(masked.contains("<RAND>"));
        assert!(masked.contains("stage=deploy"));
        assert!(!masked.contains("RjyJqt"));
    }

    #[test]
    fn signal_ratio_high_for_structured_line() {
        let line = "2025-12-29T07:58:16 MEDIUM Jenkins pipeline=pipe_2032 repo=repo_469 \
            branch=release lang=Python os=windows-latest cloud=On-Prem stage=deploy \
            failure_type=Security Scan Failure error_code=ERR_621 retry=3 flaky=True \
            msg=\"ERROR: RjyJqtFYmKiXBA5qwUE5HeQgJ2AOHlTqsFEGfE3Z\"";
        assert!(signal_ratio(line) > 0.45);
        assert!(!is_low_signal_default(line));
    }

    #[test]
    fn signal_ratio_low_for_pure_blob_line() {
        let line = "ERROR: hqMA49ZWCXsrHsXy81t5pCoPJBer4QpdnJdelM7WPCykUZENL6T469HHsAybKVxlk\
            1JyuWWQLf3x3eRPFkiO8LKPIyMEfFtC4iKqVTzy9nAKMz0ziBAIFCXD";
        assert!(signal_ratio(line) < 0.15);
        assert!(is_low_signal_default(line));
    }

    #[test]
    fn short_lines_not_judged_as_trash() {
        assert!(!is_low_signal_default("200"));
        assert!(!is_low_signal_default("OK"));
        assert!(!is_low_signal_default("x7y9"));
    }

    #[test]
    fn signal_ratio_empty_and_symbolic() {
        assert_eq!(signal_ratio(""), 1.0);
        assert_eq!(signal_ratio("--- === >>>"), 1.0);
    }

    #[test]
    fn entropy_and_case_helpers_edge_cases() {
        assert_eq!(entropy(""), 0.0);
        assert_eq!(case_transition_rate("a"), 0.0);
        assert!(!is_random_token("123456789012"));
    }

    #[test]
    fn template_key_cached_matches_uncached() {
        let line = "MEDIUM Jenkins stage=deploy error_code=ERR_621";
        assert_eq!(template_key_cached(line), template_key(line));
    }
}
