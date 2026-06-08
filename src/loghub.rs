//! Build answer-survival triples from LogHub-2.0 prose log messages.
//!
//! Targets genuinely UNSTRUCTURED prose logs, where the fact that matters is
//! stated in free narrative text with no salient key/regex anchor. Mines such
//! needles from LogHub-2.0 so the benchmark can measure whether keep-severity +
//! the structural floor preserve them (low floor survival here is the evidence
//! these are the floor's blind spot — the justification for a neural model).
//! Ported from `polymorph_lamr.bench.loghub`.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};
use serde::Serialize;
use std::path::Path;

use crate::triples::AnswerTriple;

// Floor-lockable patterns: a needle matching ANY is not a blind spot, so reject.
static FLOOR_LOCKABLE: Lazy<Vec<Regex>> = Lazy::new(|| {
    let ci = |p: &str| RegexBuilder::new(p).case_insensitive(true).build().unwrap();
    vec![
        Regex::new(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b").unwrap(),
        Regex::new(r"\b(?:FATAL|CRITICAL|ERROR|EXCEPTION|TRACEBACK|WARN(?:ING)?)\b").unwrap(),
        ci(r"(?:status|HTTP|code)[=:\s]+[45]\d{2}\b"),
        Regex::new(r"\berrno[=:]\s*\d+\b").unwrap(),
        ci(r"\berror code\s+(?:0x[0-9A-Fa-f]+|\d+)\b"),
        Regex::new(r"\b\d{1,3}(?:\.\d{1,3}){3}\b").unwrap(),
        Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b").unwrap(),
        Regex::new(r"\bINC\d{4,}\b").unwrap(),
        Regex::new(r"request_id[=:]\s*[A-Za-z0-9_-]+").unwrap(),
        ci(r#"\b(?:root_cause|resolution|resolution_action|remediation|failure_reason|reason|msg|message|short_description|summary)\s*[=:]\s*"[^"\n]{1,200}""#),
    ]
});

fn floor_would_lock(text: &str) -> bool {
    FLOOR_LOCKABLE.iter().any(|p| p.is_match(text))
}

// Per-system header strippers.
static OS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\S+\s+\d{4}-\d{2}-\d{2}\s+[\d:.]+\s+\d+\s+[A-Z]+\s+[\w.$]+\s+(?:\[[^\]]*\]\s+)?(.*)$").unwrap()
});
static SPARK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\d\d/\d\d/\d\d \d\d:\d\d:\d\d [A-Z]+ +[\w.$]+: (.*)$").unwrap());
static ZK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[\d\- :,]+ - [A-Z]+ +\[[^\]]*\] - (.*)$").unwrap());
static HADOOP_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2} [\d:,]+ [A-Z]+ +\[[^\]]*\] [\w.$]+: (.*)$").unwrap());
static HDFS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\d{6} \d{6} \d+ [A-Z]+ [\w.$]+: (.*)$").unwrap());
static LINUX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Z][a-z]{2} +\d+ [\d:]+ \S+ \S+: (.*)$").unwrap());
static BGL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\S+ \d+ [\d.]+ \S+ [\d.\-]+ \S+ RAS \w+ [A-Z]+ (.*)$").unwrap());

/// The LogHub systems we mine, in fixed order.
pub const SYSTEMS: &[&str] = &[
    "openstack", "spark", "hadoop", "zookeeper", "hdfs", "linux", "bgl",
];

fn strip_openstack(line: &str) -> Option<String> {
    let caps = OS_RE.captures(line)?;
    let msg = caps.get(1)?.as_str().trim().to_string();
    if msg.starts_with('"') || msg.to_lowercase().starts_with("get ") || msg.contains("HTTP/1.1") {
        return None;
    }
    Some(msg)
}

fn strip_generic(rx: &Regex, line: &str) -> Option<String> {
    rx.captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
}

fn strip(system: &str, line: &str) -> Option<String> {
    match system {
        "openstack" => strip_openstack(line),
        "spark" => strip_generic(&SPARK_RE, line),
        "hadoop" => strip_generic(&HADOOP_RE, line),
        "zookeeper" => strip_generic(&ZK_RE, line),
        "hdfs" => strip_generic(&HDFS_RE, line),
        "linux" => strip_generic(&LINUX_RE, line),
        "bgl" => strip_generic(&BGL_RE, line),
        _ => None,
    }
}

static VOLATILE_TAIL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\s+(?:\d+|0x[0-9a-fA-F]+|blk_-?\d+|[\w.$]+_\d+|\S*\d{3,}\S*))+\s*$").unwrap()
});
static WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z]").unwrap());
static DIGIT_RUN_3: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d{3,}").unwrap());

const QUESTION: &str = "What operational event or condition did the log report?";

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Reduce a raw message to a stable, distinctive multi-word prose needle.
fn needle_from_message(msg: &str) -> Option<String> {
    let mut needle = msg.trim().trim_end_matches('.').to_string();
    let trimmed = VOLATILE_TAIL.replace(&needle, "").trim().to_string();
    if word_count(&trimmed) >= 3 {
        needle = trimmed;
    }
    needle = needle.trim().trim_matches(':').trim().to_string();
    if word_count(&needle) < 3 {
        return None;
    }
    if !WORD.is_match(&needle) {
        return None;
    }
    if !needle.chars().any(|c| c.is_lowercase()) {
        return None;
    }
    if DIGIT_RUN_3.is_match(&needle) {
        return None;
    }
    let len = needle.chars().count();
    if !(12..=120).contains(&len) {
        return None;
    }
    Some(needle)
}

/// Mine prose needles for one LogHub system. The needle must occur exactly once
/// in its window (unambiguous survival) and be a floor blind spot.
pub fn build_triples_for_system(
    name: &str,
    text: &str,
    window_lines: usize,
    max_triples: usize,
) -> Vec<AnswerTriple> {
    let raw_lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let source = format!("loghub2:{name}");
    let mut triples: Vec<AnswerTriple> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let n = raw_lines.len();
    let half = window_lines / 2;

    for (i, line) in raw_lines.iter().enumerate() {
        if triples.len() >= max_triples {
            break;
        }
        let msg = match strip(name, line) {
            Some(m) if !m.is_empty() => m,
            _ => continue,
        };
        if floor_would_lock(&msg) {
            continue;
        }
        let needle = match needle_from_message(&msg) {
            Some(n) => n,
            None => continue,
        };
        if floor_would_lock(&needle) {
            continue;
        }
        if seen.contains(&needle) {
            continue;
        }
        if !line.contains(&needle) {
            continue;
        }
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let chunk = raw_lines[lo..hi].join("\n");
        if chunk.matches(&needle).count() != 1 {
            continue;
        }
        seen.insert(needle.clone());
        triples.push(AnswerTriple {
            doc_id: format!("{source}#{i}"),
            text: chunk,
            question: QUESTION.to_string(),
            answer: needle,
            fact_type: format!("loghub:{name}"),
            source: source.clone(),
        });
    }
    triples
}

/// Mine across all systems present as `<system>.log` files in `raw_dir`.
pub fn build_all(raw_dir: &Path, window_lines: usize, max_per_system: usize) -> Vec<AnswerTriple> {
    let mut triples = Vec::new();
    for name in SYSTEMS {
        let path = raw_dir.join(format!("{name}.log"));
        if !path.is_file() {
            continue;
        }
        let text = match std::fs::read(&path) {
            Ok(b) => String::from_utf8_lossy(&b).into_owned(),
            Err(_) => continue,
        };
        triples.extend(build_triples_for_system(name, &text, window_lines, max_per_system));
    }
    triples
}

fn fact_counts(triples: &[AnswerTriple]) -> Vec<(String, usize)> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for t in triples {
        *counts.entry(t.fact_type.clone()).or_insert(0) += 1;
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

#[derive(Serialize)]
struct LoghubPayload<'a> {
    n: usize,
    class_counts: serde_json::Value,
    fact_type_counts: serde_json::Map<String, serde_json::Value>,
    triples: &'a [AnswerTriple],
}

/// CLI: mine prose-needle triples and write the payload JSON.
pub fn run(raw_dir: &Path, out: &Path, window_lines: usize, max_per_system: usize) -> Result<()> {
    let triples = build_all(raw_dir, window_lines, max_per_system);
    let mut ft = serde_json::Map::new();
    for (k, v) in fact_counts(&triples) {
        ft.insert(k, serde_json::json!(v));
    }
    let payload = LoghubPayload {
        n: triples.len(),
        class_counts: serde_json::json!({ "prose": triples.len() }),
        fact_type_counts: ft,
        triples: &triples,
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_string_pretty(&payload)?)?;
    println!("built {} loghub prose triples", triples.len());
    println!("wrote {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_spark_message() {
        let line = "17/06/09 20:10:40 INFO executor.Executor: Send worker leaving thread now";
        let msg = strip("spark", line).unwrap();
        assert_eq!(msg, "Send worker leaving thread now");
    }

    #[test]
    fn needle_rejects_id_heavy_and_keeps_prose() {
        assert!(needle_from_message("Send worker leaving thread now").is_some());
        // too short / not enough words
        assert!(needle_from_message("done ok").is_none());
        // embeds a long digit run -> rejected
        assert!(needle_from_message("connection failed after 12345 retries done").is_none());
        // floor-lockable severity is handled upstream, but ALL-CAPS prose has no lowercase
        assert!(needle_from_message("SEND WORKER LEAVING THREAD").is_none());
    }

    #[test]
    fn mines_prose_needle_from_spark_window() {
        let mut lines: Vec<String> = (0..10)
            .map(|i| format!("17/06/09 20:10:4{} INFO storage.BlockManager: heartbeat received", i % 10))
            .collect();
        lines.insert(
            5,
            "17/06/09 20:10:45 INFO executor.Executor: Send worker leaving thread gracefully".to_string(),
        );
        let text = lines.join("\n");
        let triples = build_triples_for_system("spark", &text, 30, 40);
        assert!(triples.iter().any(|t| t.answer == "Send worker leaving thread gracefully"));
        for t in &triples {
            assert_eq!(t.fact_type, "loghub:spark");
            assert!(t.text.contains(&t.answer));
        }
    }

    #[test]
    fn floor_lockable_message_is_rejected() {
        // an ERROR severity line is floor territory -> no triple
        let line = "17/06/09 20:10:45 INFO executor.Executor: ERROR connection refused by host";
        let text = format!("{line}\n{line}");
        let triples = build_triples_for_system("spark", &text, 30, 40);
        assert!(triples.is_empty());
    }
}
