//! Phase 0b: label-ceiling QA — the $0 kill gate for the answer-survival bet.
//!
//! The model can only ever be as good as its labels. Every training label comes
//! from the teacher's compressed text (`derive_mask` keeps the tokens the teacher
//! kept). So if the teacher already dropped an answer needle that keep-severity
//! keeps at the same compression, no model trained on these labels can beat
//! keep-severity. This module measures teacher-ceiling, label-ceiling, and the
//! iso-ratio keep-severity baseline over distilled records. Ported from
//! `polymorph_lamr.bench.label_ceiling`.

use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::align::derive_mask;
use crate::methods::{CompressionMethod, KeepSeverityHeuristic};
use crate::survival::answer_survives;
use crate::tokenizer::decode_tokens;
use crate::triples::{best_semantic_triple_for_chunk, best_triple_for_chunk};

#[derive(Debug, Default)]
pub struct CeilingCounts {
    pub chunks: usize,
    pub with_needle: usize,
    pub needle_in_original: usize,
    pub teacher_survived: usize,
    pub label_survived: usize,
    pub keepsev_survived: usize,
    pub sum_teacher_drop: f64,
    /// fact_type -> [denom, teacher_survived, keepsev_survived].
    pub by_type: BTreeMap<String, [usize; 3]>,
}

/// Reconstruct the text the trainer's keep-mask preserves, and the teacher's
/// achieved token drop rate.
fn kept_text(original: &str, compressed: &str) -> Result<(String, f64)> {
    let (ids, _spans, keep) = derive_mask(original, compressed)?;
    let kept_ids: Vec<u32> = ids
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(id, _)| *id)
        .collect();
    let n = ids.len();
    let drop = if n > 0 {
        1.0 - (kept_ids.len() as f64 / n as f64)
    } else {
        0.0
    };
    Ok((decode_tokens(&kept_ids)?, drop))
}

/// Yield `(original, compressed, src_path)` from a distilled JSONL, skipping
/// blanks / malformed / empty-pair lines.
pub fn iter_records(content: &str, limit: Option<usize>) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let original = rec.get("original").and_then(|v| v.as_str()).unwrap_or("");
        let compressed = rec.get("compressed").and_then(|v| v.as_str()).unwrap_or("");
        if original.is_empty() || compressed.is_empty() {
            continue;
        }
        let src = rec
            .get("src_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.push((original.to_string(), compressed.to_string(), src));
        if let Some(lim) = limit {
            if out.len() >= lim {
                break;
            }
        }
    }
    out
}

pub fn measure(content: &str, limit: Option<usize>, semantic: bool) -> Result<CeilingCounts> {
    let mut c = CeilingCounts::default();
    let keepsev = KeepSeverityHeuristic;
    for (ci, (original, compressed, source)) in iter_records(content, limit).into_iter().enumerate() {
        c.chunks += 1;
        let doc_id = format!("{source}#{ci}");
        let triple = if semantic {
            best_semantic_triple_for_chunk(&doc_id, &original, &source)
        } else {
            best_triple_for_chunk(&doc_id, &original, &source)
        };
        let triple = match triple {
            Some(t) => t,
            None => continue,
        };
        c.with_needle += 1;
        let needle = &triple.answer;
        if !answer_survives(needle, &original) {
            continue;
        }
        c.needle_in_original += 1;
        let bucket = c.by_type.entry(triple.fact_type.clone()).or_insert([0, 0, 0]);
        bucket[0] += 1;

        if answer_survives(needle, &compressed) {
            c.teacher_survived += 1;
            bucket[1] += 1;
        }

        let (kept, teacher_drop) = kept_text(&original, &compressed)?;
        c.sum_teacher_drop += teacher_drop;
        if answer_survives(needle, &kept) {
            c.label_survived += 1;
        }

        let r = teacher_drop.clamp(0.0, 1.0);
        if answer_survives(needle, &keepsev.compress(&original, r)?) {
            c.keepsev_survived += 1;
            // re-borrow bucket (it was dropped across the await-free calls above)
            c.by_type.get_mut(&triple.fact_type).unwrap()[2] += 1;
        }
    }
    Ok(c)
}

fn pct(num: usize, den: usize) -> String {
    if den > 0 {
        format!("{:.1}%", 100.0 * num as f64 / den as f64)
    } else {
        "n/a".to_string()
    }
}

pub fn format_report(c: &CeilingCounts) -> String {
    let den = c.needle_in_original;
    let mean_drop = if den > 0 {
        c.sum_teacher_drop / den as f64
    } else {
        0.0
    };
    let mut lines = vec![
        "== LaMR label-ceiling QA (Phase 0b) ==".to_string(),
        format!("records scanned        : {}", c.chunks),
        format!("with mineable needle   : {}", c.with_needle),
        format!(
            "needle present in orig : {}  (the survival denominator)",
            c.needle_in_original
        ),
        format!("teacher mean drop rate : {mean_drop:.3}"),
        String::new(),
        format!(
            "teacher-ceiling survival : {}  ({}/{})  <- the hard cap on any model",
            pct(c.teacher_survived, den),
            c.teacher_survived,
            den
        ),
        format!(
            "label-ceiling survival   : {}  ({}/{})  <- what the trainer actually learns",
            pct(c.label_survived, den),
            c.label_survived,
            den
        ),
        format!(
            "keep-severity @ same drop: {}  ({}/{})  <- the baseline to beat",
            pct(c.keepsev_survived, den),
            c.keepsev_survived,
            den
        ),
        String::new(),
        "per-fact-type survival (teacher vs keep-severity, n):".to_string(),
    ];
    let mut by: Vec<(&String, &[usize; 3])> = c.by_type.iter().collect();
    by.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
    for (ftype, counts) in by {
        let [n, tsurv, ksurv] = *counts;
        lines.push(format!(
            "  {:<12} teacher {:>6}  keep-sev {:>6}  (n={})",
            ftype,
            pct(tsurv, n),
            pct(ksurv, n),
            n
        ));
    }
    lines.push(String::new());
    if den > 0 {
        let teacher = c.teacher_survived as f64 / den as f64;
        let baseline = c.keepsev_survived as f64 / den as f64;
        if teacher < baseline - 1e-9 {
            lines.push(
                "VERDICT: GATE FAILS. Teacher labels preserve FEWER needles than \
                 keep-severity at the same compression. No model trained on these \
                 labels can beat the baseline. Fix labels before spending GPU credit."
                    .to_string(),
            );
        } else if teacher < baseline + 1e-9 {
            lines.push(
                "VERDICT: MARGINAL. Teacher labels only match keep-severity. The \
                 model's edge must come entirely from sub-line precision. Proceed, \
                 but expect a thin win."
                    .to_string(),
            );
        } else {
            lines.push(format!(
                "VERDICT: GATE CLEARS (label side). Teacher labels preserve MORE \
                 needles than keep-severity (headroom {:.1} pts). The bet is live.",
                100.0 * (teacher - baseline)
            ));
        }
    }
    lines.join("\n")
}

/// CLI: read a distilled JSONL, measure, print the report, optionally dump JSON.
pub fn run(distilled: &Path, limit: Option<usize>, semantic: bool, out: Option<&Path>) -> Result<()> {
    if !distilled.is_file() {
        anyhow::bail!("distilled file not found: {}", distilled.display());
    }
    let content = std::fs::read_to_string(distilled)?;
    let c = measure(&content, limit, semantic)?;
    println!("{}", format_report(&c));
    if let Some(out) = out {
        let mut by_type = serde_json::Map::new();
        for (ft, [n, t, k]) in &c.by_type {
            by_type.insert(
                ft.clone(),
                serde_json::json!({ "n": n, "teacher_survived": t, "keepsev_survived": k }),
            );
        }
        let payload = serde_json::json!({
            "chunks": c.chunks,
            "with_needle": c.with_needle,
            "needle_in_original": c.needle_in_original,
            "teacher_survived": c.teacher_survived,
            "label_survived": c.label_survived,
            "keepsev_survived": c.keepsev_survived,
            "sum_teacher_drop": c.sum_teacher_drop,
            "by_type": by_type,
        });
        std::fs::write(out, serde_json::to_string_pretty(&payload)?)?;
        println!("\nwrote {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_teacher_and_label_survival() {
        // One record: original has a unique request_id needle; compressed keeps it.
        let original = "2023-01-01 INFO heartbeat ok\n\
            2023-01-01 ERROR boom request_id=ABC123 client_ip=10.0.0.9";
        let compressed = "ERROR boom request_id=ABC123";
        let rec = serde_json::json!({
            "original": original,
            "compressed": compressed,
            "src_path": "t.log",
        });
        let content = rec.to_string();
        let c = measure(&content, None, false).unwrap();
        assert_eq!(c.chunks, 1);
        assert_eq!(c.with_needle, 1);
        assert_eq!(c.needle_in_original, 1);
        // teacher kept request_id=ABC123 -> needle survives
        assert_eq!(c.teacher_survived, 1);
        assert_eq!(c.label_survived, 1);
        let report = format_report(&c);
        assert!(report.contains("label-ceiling survival"));
    }

    #[test]
    fn dropped_needle_is_not_counted_as_survived() {
        let original = "2023-01-01 ERROR boom request_id=ABC123 client_ip=10.0.0.9";
        // compressed drops the needle entirely
        let compressed = "ERROR boom";
        let content = serde_json::json!({
            "original": original, "compressed": compressed, "src_path": "t"
        })
        .to_string();
        let c = measure(&content, None, false).unwrap();
        assert_eq!(c.needle_in_original, 1);
        assert_eq!(c.teacher_survived, 0);
    }

    #[test]
    fn skips_blank_and_malformed_lines() {
        let content = "\n  \nnot json\n{\"original\":\"\",\"compressed\":\"x\"}\n";
        let recs = iter_records(content, None);
        assert!(recs.is_empty());
    }
}
