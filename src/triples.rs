//! (log, question, answer) triple generation for the answer-survival benchmark.
//!
//! The benchmark asks the real question a log compressor must answer: *after*
//! you compress a chunk of logs, can a downstream reader still recover the fact
//! that mattered? "The fact that mattered" is operationalized as an **answer
//! needle** — a rare, salient substring (an exception type, an HTTP status, a
//! request id, an incident number, a client IP, a severity) extracted by regex
//! and paired with a templated extraction question.
//!
//! GPU-free and deterministic: extraction is pure regex, no model, no API.
//! Ported from the Python `polymorph_lamr.bench.triples`.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerTriple {
    pub doc_id: String,
    pub text: String,
    pub question: String,
    pub answer: String,
    pub fact_type: String,
    pub source: String,
}

// ---------------------------------------------------------------------------
// Structural extractors — (fact_type, regex, capture-group index, question).
// `grp == 0` means "first non-None group" (an alternation). Higher-priority
// fact types are listed first.
// ---------------------------------------------------------------------------

struct Extractor {
    fact_type: &'static str,
    re: Regex,
    grp: usize,
    question: &'static str,
}

static EXTRACTORS: Lazy<Vec<Extractor>> = Lazy::new(|| {
    let ci = |p: &str| RegexBuilder::new(p).case_insensitive(true).build().unwrap();
    vec![
        Extractor {
            fact_type: "exception",
            re: Regex::new(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b").unwrap(),
            grp: 1,
            question: "Which exception type was raised?",
        },
        Extractor {
            fact_type: "incident",
            re: Regex::new(r"\b(INC\d{4,})\b").unwrap(),
            grp: 1,
            question: "What incident number is referenced?",
        },
        Extractor {
            fact_type: "request_id",
            re: Regex::new(r"request_id[=:]\s*([A-Za-z0-9_-]+)").unwrap(),
            grp: 1,
            question: "What was the request_id of the affected request?",
        },
        Extractor {
            fact_type: "http_status",
            re: ci(r"\b(?:status|HTTP|code)[=:\s]+([45]\d{2})\b"),
            grp: 1,
            question: "What HTTP status code was returned?",
        },
        Extractor {
            fact_type: "error_code",
            re: Regex::new(r"\berrno[=:]\s*(\d+)\b|\berror code\s+(0x[0-9A-Fa-f]+|\d+)\b").unwrap(),
            grp: 0,
            question: "What error code was reported?",
        },
        Extractor {
            fact_type: "client_ip",
            re: Regex::new(r"client_ip[=:]\s*(\d{1,3}(?:\.\d{1,3}){3})").unwrap(),
            grp: 1,
            question: "What client IP is associated with the event?",
        },
        Extractor {
            fact_type: "severity",
            re: Regex::new(r"\b(FATAL|CRITICAL|ERROR)\b").unwrap(),
            grp: 1,
            question: "What is the most severe log level present in this chunk?",
        },
    ]
});

// Semantic extractors — free-text VALUES of salient keys. The order of keys in
// the alternation matches the Python dict insertion order.
const SEMANTIC_KEYS: &[(&str, &str)] = &[
    ("root_cause", "What was the root cause?"),
    ("resolution", "What resolution or action was applied?"),
    (
        "resolution_action",
        "What resolution or action was applied?",
    ),
    ("remediation", "What remediation was applied?"),
    ("reason", "What reason was given?"),
    ("failure_reason", "What was the failure reason?"),
    ("msg", "What did the log message say?"),
    ("message", "What did the log message say?"),
    ("short_description", "What was the issue described as?"),
    ("summary", "What was the summary?"),
];

static SEMANTIC_QUOTED: Lazy<Regex> = Lazy::new(|| {
    let keys = SEMANTIC_KEYS
        .iter()
        .map(|(k, _)| *k)
        .collect::<Vec<_>>()
        .join("|");
    RegexBuilder::new(&format!(r#"\b({keys})\s*[=:]\s*"([^"\n]{{3,}})""#))
        .case_insensitive(true)
        .build()
        .unwrap()
});

fn semantic_question(key: &str) -> &'static str {
    SEMANTIC_KEYS
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, q)| *q)
        .unwrap_or("What was reported?")
}

/// `(fact_type, question, answer)` candidates, in priority order.
fn candidates(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for ex in EXTRACTORS.iter() {
        for caps in ex.re.captures_iter(text) {
            let answer = if ex.grp == 0 {
                // alternation: first non-None group
                (1..caps.len()).find_map(|i| caps.get(i).map(|m| m.as_str().to_string()))
            } else {
                caps.get(ex.grp).map(|m| m.as_str().to_string())
            };
            if let Some(answer) = answer {
                if answer.len() >= 2 {
                    out.push((ex.fact_type.to_string(), ex.question.to_string(), answer));
                }
            }
        }
    }
    out
}

/// Free-text field values as `(fact_type="semantic:<key>", question, answer)`.
fn semantic_candidates(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for caps in SEMANTIC_QUOTED.captures_iter(text) {
        let key = caps.get(1).unwrap().as_str().to_lowercase();
        let value = caps.get(2).unwrap().as_str().trim().to_string();
        if !value.contains(' ') {
            continue; // require a real phrase, not a single token
        }
        let question = semantic_question(&key);
        out.push((format!("semantic:{key}"), question.to_string(), value));
    }
    out
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

fn best_from_candidates(
    doc_id: &str,
    text: &str,
    source: &str,
    cands: Vec<(String, String, String)>,
) -> Option<AnswerTriple> {
    let mut seen_types: Vec<String> = Vec::new();
    for (fact_type, question, answer) in &cands {
        if seen_types.contains(fact_type) {
            continue;
        }
        seen_types.push(fact_type.clone());
        if count_occurrences(text, answer) == 1 {
            return Some(AnswerTriple {
                doc_id: doc_id.to_string(),
                text: text.to_string(),
                question: question.clone(),
                answer: answer.clone(),
                fact_type: fact_type.clone(),
                source: source.to_string(),
            });
        }
    }
    // Fall back to the first candidate even if it repeats.
    cands
        .first()
        .map(|(fact_type, question, answer)| AnswerTriple {
            doc_id: doc_id.to_string(),
            text: text.to_string(),
            question: question.clone(),
            answer: answer.clone(),
            fact_type: fact_type.clone(),
            source: source.to_string(),
        })
}

/// Pick the highest-priority structural candidate whose answer is unique.
pub fn best_triple_for_chunk(doc_id: &str, text: &str, source: &str) -> Option<AnswerTriple> {
    best_from_candidates(doc_id, text, source, candidates(text))
}

/// Pick a unique multi-word semantic value as the needle (regex-floor-proof).
pub fn best_semantic_triple_for_chunk(
    doc_id: &str,
    text: &str,
    source: &str,
) -> Option<AnswerTriple> {
    best_from_candidates(doc_id, text, source, semantic_candidates(text))
}

fn line_windows(lines: &[String], window: usize, stride: usize) -> Vec<Vec<String>> {
    let n = lines.len();
    let mut out = Vec::new();
    if n <= window {
        if !lines.is_empty() {
            out.push(lines.to_vec());
        }
        return out;
    }
    let mut i = 0;
    while i < n {
        let end = (i + window).min(n);
        let chunk = lines[i..end].to_vec();
        if !chunk.is_empty() {
            out.push(chunk);
        }
        if i + window >= n {
            return out;
        }
        i += stride;
    }
    out
}

/// Mine answer triples from a block of log text by line-windowing.
pub fn build_triples_from_text(
    text: &str,
    source: &str,
    window_lines: usize,
    stride: Option<usize>,
    max_chunks: Option<usize>,
) -> Vec<AnswerTriple> {
    let stride = stride.unwrap_or(window_lines).max(1);
    let lines: Vec<String> = text
        .lines()
        .filter(|ln| !ln.trim().is_empty())
        .map(|ln| ln.to_string())
        .collect();
    let mut triples = Vec::new();
    for (ci, chunk_lines) in line_windows(&lines, window_lines, stride)
        .into_iter()
        .enumerate()
    {
        if let Some(mc) = max_chunks {
            if ci >= mc {
                break;
            }
        }
        let chunk = chunk_lines.join("\n");
        if let Some(t) = best_triple_for_chunk(&format!("{source}#{ci}"), &chunk, source) {
            triples.push(t);
        }
    }
    triples
}

/// Mine answer triples from a set of log files.
pub fn build_triples_from_paths(
    paths: &[PathBuf],
    window_lines: usize,
    max_per_file: usize,
    max_total: Option<usize>,
    max_bytes_per_file: usize,
) -> Vec<AnswerTriple> {
    let mut triples: Vec<AnswerTriple> = Vec::new();
    for path in paths {
        if let Some(mt) = max_total {
            if triples.len() >= mt {
                break;
            }
        }
        let raw = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let truncated = &raw[..raw.len().min(max_bytes_per_file)];
        let text = String::from_utf8_lossy(truncated);
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let got = build_triples_from_text(&text, &name, window_lines, None, None);
        triples.extend(got.into_iter().take(max_per_file));
    }
    if let Some(mt) = max_total {
        triples.truncate(mt);
    }
    triples
}

/// Recursively collect log-like files under `root` (skipping sidecars).
pub fn collect_log_files(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(_) => return,
        };
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out);
            } else if p.is_file() {
                let ext = p
                    .extension()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if matches!(ext.as_str(), "log" | "txt" | "json" | "jsonl") {
                    let name = p
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if name.ends_with(".survive") || name.starts_with("potentialAnomalies") {
                        continue;
                    }
                    out.push(p);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Curated fixtures — small, self-contained triples for tests (no data/ dep).
// ---------------------------------------------------------------------------

fn fixture_docs() -> Vec<(String, String)> {
    let distsys = {
        let mut lines: Vec<String> = (0..30)
            .map(|i| {
                format!(
                    "2023-11-20T08:40:5{}.000 INFO [ServiceA] heartbeat ok request_id=10{:02} client_ip=10.0.0.{}",
                    i % 10,
                    i,
                    i % 5
                )
            })
            .collect();
        let mut text = lines.join("\n");
        text.push_str(
            "\n2023-11-20T08:41:02.111 FATAL [ServiceB] Crash request_id=99042 client_ip=192.168.1.185 time_taken=72ms",
        );
        lines.clear();
        text
    };
    let traceback = "Traceback (most recent call last): File \"app.py\", line 42, in run\n  File \"db.py\", line 10, in connect\nConnectionResetError: connection reset by peer".to_string();
    let servicenow = (0..5)
        .map(|i| {
            format!(
                "29/2/2016 0{i}:23 incident=INC000004{i} state=New priority=3 category=Category 55"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let api = {
        let mut text = (0..20)
            .map(|i| format!("GET /v1/items 200 ok latency=12ms id=req{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        text.push_str("\nPOST /v1/checkout HTTP status=503 service unavailable id=reqX9 retries=3");
        text
    };
    vec![
        ("distsys".to_string(), distsys),
        ("traceback".to_string(), traceback),
        ("servicenow".to_string(), servicenow),
        ("api".to_string(), api),
    ]
}

/// Deterministic, dependency-free triples for tests.
pub fn curated_triples() -> Vec<AnswerTriple> {
    let mut out = Vec::new();
    for (source, text) in fixture_docs() {
        if let Some(t) = best_triple_for_chunk(&format!("{source}#0"), &text, &source) {
            out.push(t);
        }
    }
    out
}

/// JSON dump helper used by the CLI subcommand.
pub fn dump_json(triples: &[AnswerTriple]) -> Result<String> {
    Ok(serde_json::to_string_pretty(triples)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_triples_have_answers_in_text() {
        let triples = curated_triples();
        assert_eq!(triples.len(), 4);
        for t in &triples {
            assert!(!t.answer.is_empty() && t.text.contains(&t.answer));
            assert!(t.question.ends_with('?'));
            assert!(!t.fact_type.is_empty());
        }
    }

    #[test]
    fn curated_triple_fact_types_are_deterministic() {
        let triples = curated_triples();
        let types: Vec<&str> = triples.iter().map(|t| t.fact_type.as_str()).collect();
        assert_eq!(
            types,
            vec!["request_id", "exception", "incident", "http_status"]
        );
        assert_eq!(triples[0].answer, "1000");
        assert_eq!(triples[1].answer, "ConnectionResetError");
        assert_eq!(triples[2].answer, "INC0000040");
        assert_eq!(triples[3].answer, "503");
    }

    #[test]
    fn build_triples_extracts_a_needle() {
        let mut lines: Vec<String> = vec!["2023-01-01 INFO heartbeat ok".to_string(); 10];
        lines.push("2023-01-01 ERROR boom request_id=ABC123 client_ip=10.0.0.9".to_string());
        let text = lines.join("\n");
        let triples = build_triples_from_text(&text, "t", 50, None, None);
        assert!(!triples.is_empty());
        assert!(triples[0].text.contains(&triples[0].answer));
    }

    #[test]
    fn semantic_extractor_pulls_freetext_phrase() {
        let text = "ERROR payment-api status=503 client_ip=10.0.0.9 \
            msg=\"Internal server error\" root_cause=\"Memory exhaustion here\" resolution=\"Restart\"";
        let t = best_semantic_triple_for_chunk("d#0", text, "s").unwrap();
        assert!(t.fact_type.starts_with("semantic:"));
        assert!(t.answer.contains(' '));
    }
}
