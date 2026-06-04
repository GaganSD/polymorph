//! Compression benchmark harness (the evidence gate for direction B).
//!
//! Measures the DETERMINISTIC stack on real corpora and reports the numbers that
//! decide whether a neural pruner is ever worth building:
//!   - **compression ratio** (cl100k tokens in / out) from the dedup pre-stage,
//!   - **throughput** (MB/s) and **per-chunk p50/p95/p99 latency** (the SLO is
//!     throughput + per-chunk p95, NOT a flat 150ms — a 10MB blob is ~2.5M tokens),
//!   - an **answer-token survival proxy**: fraction of seeded must-survive strings
//!     still present after compression. This is the cheap stand-in for downstream
//!     answer-accuracy until the full trace-QA eval set lands (see TODOS.md). A high
//!     ratio with low survival is a RED flag, not a win.
//!
//! Run: `polymorph-mcp --bench <corpus_dir> [chunk_kb] [max_mb]`
//!
//! Survival tokens: a sibling `<file>.survive` (one substring per line) is used if
//! present; otherwise, for TrainTicket, any sibling `potentialAnomalies_*.txt` in
//! the same directory seeds must-survive strings. Without either, survival is
//! reported as n/a.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dedup::{dedup_plan, DedupOpts};
use crate::tokenizer;

const DEFAULT_CHUNK_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_BYTES: usize = 25 * 1024 * 1024; // cap per file so the bench stays snappy
const TOTAL_BUDGET_BYTES: usize = 64 * 1024 * 1024; // global cap across all files (bounded run)

struct CorpusStats {
    name: String,
    files: usize,
    bytes_processed: usize,
    orig_tokens: usize,
    reduced_tokens: usize,
    elided_lines: usize,
    chunk_latencies_ms: Vec<f64>,
    total_secs: f64,
    survive_total: usize,
    survive_kept: usize,
}

/// Entry point for `--bench`.
pub fn run(corpus_dir: &str, chunk_kb: Option<usize>, max_mb: Option<usize>) -> Result<()> {
    let chunk_bytes = chunk_kb.map(|k| k * 1024).unwrap_or(DEFAULT_CHUNK_BYTES);
    let max_bytes = max_mb
        .map(|m| m * 1024 * 1024)
        .unwrap_or(DEFAULT_MAX_BYTES);
    let root = Path::new(corpus_dir);
    if !root.exists() {
        anyhow::bail!("corpus dir not found: {corpus_dir}");
    }

    let files = collect_files(root);
    if files.is_empty() {
        anyhow::bail!("no .log/.txt/.json files under {corpus_dir}");
    }

    println!("polymorph benchmark");
    println!("corpus: {corpus_dir}");
    println!(
        "chunk={} KB  max/file={} MB  files={}\n",
        chunk_bytes / 1024,
        max_bytes / (1024 * 1024),
        files.len()
    );

    let mut stats = CorpusStats {
        name: corpus_dir.to_string(),
        files: 0,
        bytes_processed: 0,
        orig_tokens: 0,
        reduced_tokens: 0,
        elided_lines: 0,
        chunk_latencies_ms: Vec::new(),
        total_secs: 0.0,
        survive_total: 0,
        survive_kept: 0,
    };

    let mut budget_hit = false;
    for file in &files {
        if stats.bytes_processed >= TOTAL_BUDGET_BYTES {
            budget_hit = true;
            break;
        }
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(_) => continue, // skip non-utf8 / unreadable
        };
        let remaining = TOTAL_BUDGET_BYTES.saturating_sub(stats.bytes_processed);
        let text = truncate_on_line_boundary(&text, max_bytes.min(remaining));
        let survive = load_survive_tokens(file);

        for chunk in line_chunks(text, chunk_bytes) {
            let start = Instant::now();
            let plan = dedup_plan(chunk, DedupOpts::default());
            let elapsed = start.elapsed();
            stats.chunk_latencies_ms.push(elapsed.as_secs_f64() * 1000.0);
            stats.total_secs += elapsed.as_secs_f64();
            stats.bytes_processed += chunk.len();
            stats.orig_tokens += tokenizer::count_tokens(chunk).unwrap_or(0);
            stats.reduced_tokens += tokenizer::count_tokens(&plan.reduced).unwrap_or(0);
            stats.elided_lines += plan.elided_line_count();
            // survival proxy on the reduced text
            for tok in &survive {
                stats.survive_total += 1;
                if plan.reduced.contains(tok.as_str()) {
                    stats.survive_kept += 1;
                }
            }
        }
        stats.files += 1;
    }

    report(&stats);
    if budget_hit {
        println!(
            "(stopped at the {} MB global budget; pass a smaller corpus or subdir for full coverage)",
            TOTAL_BUDGET_BYTES / (1024 * 1024)
        );
    }
    Ok(())
}

fn report(s: &CorpusStats) {
    let mut lat = s.chunk_latencies_ms.clone();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        if lat.is_empty() {
            return 0.0;
        }
        let idx = ((p / 100.0) * (lat.len() as f64 - 1.0)).round() as usize;
        lat[idx.min(lat.len() - 1)]
    };
    let ratio = if s.reduced_tokens > 0 {
        s.orig_tokens as f64 / s.reduced_tokens as f64
    } else {
        0.0
    };
    let mb = s.bytes_processed as f64 / (1024.0 * 1024.0);
    let mbps = if s.total_secs > 0.0 {
        mb / s.total_secs
    } else {
        0.0
    };

    println!("== results: {} ==", s.name);
    println!("files processed       : {}", s.files);
    println!("bytes processed       : {:.1} MB", mb);
    println!("chunks                : {}", lat.len());
    println!(
        "tokens (cl100k)       : {} -> {}",
        s.orig_tokens, s.reduced_tokens
    );
    println!(
        "compression ratio     : {:.2}x  ({:.1}% tokens removed)",
        ratio,
        if s.orig_tokens > 0 {
            100.0 * (1.0 - s.reduced_tokens as f64 / s.orig_tokens as f64)
        } else {
            0.0
        }
    );
    println!("lines elided          : {}", s.elided_lines);
    println!("throughput            : {:.1} MB/s", mbps);
    println!(
        "per-chunk latency ms  : p50={:.2}  p95={:.2}  p99={:.2}",
        pct(50.0),
        pct(95.0),
        pct(99.0)
    );
    if s.survive_total > 0 {
        let surv = 100.0 * s.survive_kept as f64 / s.survive_total as f64;
        let flag = if surv < 99.9 { "  <-- CHECK: answer tokens lost" } else { "" };
        println!(
            "answer-token survival : {:.2}% ({}/{}){}",
            surv, s.survive_kept, s.survive_total, flag
        );
    } else {
        println!("answer-token survival : n/a (no .survive / potentialAnomalies sidecar)");
    }
    println!(
        "\nNOTE: ratio is the deterministic dedup gain only. Answer-accuracy is a\n\
         keep-the-tokens proxy until the trace-QA eval set lands (TODOS.md, P1).\n\
         SLO is throughput + per-chunk p95, not a flat 150ms (a 10MB trace ~2.5M tokens)."
    );
}

/// Split `text` into chunks of ~`chunk_bytes`, always breaking on a line boundary
/// so a chunk is never split mid-record.
fn line_chunks(text: &str, chunk_bytes: usize) -> Vec<&str> {
    if text.len() <= chunk_bytes {
        return vec![text];
    }
    let bytes = text.as_bytes();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_bytes).min(text.len());
        if end < text.len() {
            // walk back to the last newline within the window
            match bytes[start..end].iter().rposition(|&b| b == b'\n') {
                Some(pos) => end = start + pos + 1,
                // A single line longer than the window: emit it whole up to a
                // char boundary so we never slice mid-UTF-8-codepoint (panic).
                // Rounding UP guarantees end > start, so the loop makes progress.
                None => end = ceil_char_boundary(text, end),
            }
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

fn truncate_on_line_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let bytes = text.as_bytes();
    match bytes[..max_bytes].iter().rposition(|&b| b == b'\n') {
        Some(pos) => &text[..pos + 1],
        // No newline in the window: truncate at a char boundary at-or-below
        // max_bytes so we never slice mid-UTF-8-codepoint (panic).
        None => &text[..floor_char_boundary(text, max_bytes)],
    }
}

/// Smallest char boundary >= `idx` (clamped to len). Rounds a byte offset UP to
/// the next valid UTF-8 boundary.
fn ceil_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

/// Largest char boundary <= `idx`. Rounds a byte offset DOWN to a valid boundary.
fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_files_inner(root, &mut out);
    out.sort();
    out
}

fn collect_files_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_inner(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("log") | Some("txt") | Some("json")
        ) {
            // skip our own sidecars / parser artifacts
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".survive") || name.starts_with("potentialAnomalies") {
                continue;
            }
            out.push(path);
        }
    }
}

/// Load must-survive substrings: prefer a sibling `<file>.survive` (one per line);
/// else seed from any `potentialAnomalies_*.txt` in the same directory (TrainTicket).
fn load_survive_tokens(file: &Path) -> Vec<String> {
    let sidecar = file.with_extension(format!(
        "{}.survive",
        file.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    if let Ok(content) = std::fs::read_to_string(&sidecar) {
        return content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
    }
    // Fall back to TrainTicket potentialAnomalies in the same dir.
    if let Some(dir) = file.parent() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let name = e.file_name();
                let name = name.to_str().unwrap_or("");
                if name.starts_with("potentialAnomalies") && name.ends_with(".txt") {
                    if let Ok(content) = std::fs::read_to_string(e.path())
                        .with_context(|| format!("reading {name}"))
                    {
                        return content
                            .lines()
                            .map(|l| l.trim().to_string())
                            .filter(|l| l.len() > 8) // skip trivially-short noise
                            .take(50)
                            .collect();
                    }
                }
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_chunks_break_on_newline() {
        let text = "aaaa\nbbbb\ncccc\ndddd\n";
        let chunks = line_chunks(text, 6);
        // every chunk (except possibly a lone overlong line) ends on a newline
        for c in &chunks[..chunks.len().saturating_sub(1)] {
            assert!(c.ends_with('\n'));
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn truncate_keeps_line_boundary() {
        let text = "line1\nline2\nline3\n";
        let t = truncate_on_line_boundary(text, 8);
        assert!(t.ends_with('\n'));
        assert!(text.starts_with(t));
    }

    #[test]
    fn single_chunk_when_small() {
        assert_eq!(line_chunks("tiny", 1024), vec!["tiny"]);
    }

    #[test]
    fn line_chunks_multibyte_no_newline_does_not_panic() {
        // Regression: a single line longer than the window with a multibyte char
        // straddling the byte cut must not panic, and chunks must reassemble.
        let text = "aa\u{1F4A5}\u{1F4A5}"; // "aa💥💥", 10 bytes, no newline
        let chunks = line_chunks(text, 4);
        assert_eq!(chunks.concat(), text);
        for c in &chunks {
            assert!(c.is_char_boundary(0)); // each chunk is a valid &str (no mid-codepoint)
        }
    }

    #[test]
    fn line_chunks_multibyte_with_newlines() {
        let text = "héllo 🌍\nwörld 🎉\n";
        let chunks = line_chunks(text, 8);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn truncate_multibyte_no_newline_does_not_panic() {
        let text = "aa\u{1F4A5}\u{1F4A5}"; // max_bytes lands mid-💥
        let t = truncate_on_line_boundary(text, 4);
        assert!(text.starts_with(t));
        assert_eq!(t, "aa"); // floored to the boundary before the first 💥
    }

    #[test]
    fn char_boundary_helpers() {
        let s = "a\u{1F4A5}b"; // a 💥 b : bytes 0,1..5,5
        assert_eq!(ceil_char_boundary(s, 2), 5);
        assert_eq!(floor_char_boundary(s, 2), 1);
        assert_eq!(ceil_char_boundary(s, 100), s.len());
    }

    use std::io::Write;

    #[test]
    fn run_on_temp_corpus_processes_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("app.log")).unwrap();
        for _ in 0..50 {
            writeln!(f, "2026-01-01T00:00:00Z INFO heartbeat ok id=1").unwrap();
        }
        writeln!(f, "ERROR 500 boom").unwrap();
        // survival sidecar (exercises the sidecar branch + survival reporting)
        std::fs::write(dir.path().join("app.log.survive"), "ERROR 500 boom\n").unwrap();
        let res = run(dir.path().to_str().unwrap(), Some(64), Some(1));
        assert!(res.is_ok());
    }

    #[test]
    fn run_errors_on_missing_dir() {
        assert!(run("/nonexistent/polymorph/xyz", None, None).is_err());
    }

    #[test]
    fn run_errors_on_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(run(dir.path().to_str().unwrap(), None, None).is_err());
    }

    #[test]
    fn collect_files_filters_sidecars_and_anomalies() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.log"), "x").unwrap();
        std::fs::write(dir.path().join("a.log.survive"), "x").unwrap();
        std::fs::write(dir.path().join("potentialAnomalies_x.txt"), "x").unwrap();
        std::fs::write(dir.path().join("b.json"), "{}").unwrap();
        std::fs::write(dir.path().join("c.csv"), "x").unwrap(); // ignored extension
        let names: Vec<String> = collect_files(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert!(names.contains(&"a.log".to_string()));
        assert!(names.contains(&"b.json".to_string()));
        assert!(!names.iter().any(|n| n.ends_with(".survive")));
        assert!(!names.iter().any(|n| n.starts_with("potentialAnomalies")));
        assert!(!names.contains(&"c.csv".to_string()));
    }

    #[test]
    fn load_survive_tokens_sidecar_and_anomalies_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let logf = dir.path().join("x.log");
        std::fs::write(&logf, "data").unwrap();
        // nothing yet -> empty
        assert!(load_survive_tokens(&logf).is_empty());
        // potentialAnomalies fallback (long lines only)
        std::fs::write(
            dir.path().join("potentialAnomalies_x.txt"),
            "some anomaly line here\nshort\n",
        )
        .unwrap();
        assert!(load_survive_tokens(&logf).iter().any(|t| t.contains("anomaly")));
        // sidecar takes precedence
        std::fs::write(logf.with_extension("log.survive"), "EXACT-TOKEN\n").unwrap();
        assert_eq!(load_survive_tokens(&logf), vec!["EXACT-TOKEN".to_string()]);
    }
}
