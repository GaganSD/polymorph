//! Answer-survival metric + the rate-distortion sweep and report.
//!
//! The benchmark maps **answer survival** (did the needle fact survive
//! compression?) against **compression ratio** (tokens in / tokens out) for each
//! method. The primary survival test is a GPU-free, deterministic exact-match:
//! after collapsing whitespace and case, does the answer substring still appear
//! in the compressed text? Ported from `polymorph_lamr.bench.survival` (the LLM
//! judge variant is out of scope — it needs a network model).

use once_cell::sync::Lazy;
use regex::Regex;

use crate::methods::{token_count, CompressionMethod};
use crate::stats::mcnemar_paired;
use crate::triples::AnswerTriple;

static WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

fn norm(s: &str) -> String {
    WS.replace_all(s, " ").trim().to_lowercase()
}

/// Exact-match survival: whitespace-collapsed, case-insensitive substring.
pub fn answer_survives(answer: &str, compressed: &str) -> bool {
    norm(compressed).contains(&norm(answer))
}

/// tokens_in / tokens_out (>= 1 means smaller; higher = more compression).
pub fn compression_ratio(orig_text: &str, comp_text: &str) -> f64 {
    let out = token_count(comp_text);
    if out == 0 {
        f64::INFINITY
    } else {
        token_count(orig_text) as f64 / out as f64
    }
}

/// Fraction of tokens removed.
pub fn achieved_drop_rate(orig_text: &str, comp_text: &str) -> f64 {
    let o = token_count(orig_text);
    if o == 0 {
        0.0
    } else {
        1.0 - (token_count(comp_text) as f64 / o as f64)
    }
}

#[derive(Debug, Clone)]
pub struct MethodRow {
    pub method: String,
    pub target_drop_rate: f64,
    pub survival: f64,
    pub mean_ratio: f64,
    pub mean_achieved_drop: f64,
    pub n: usize,
}

/// Default survival test: exact-match of the needle in the compressed text.
pub fn default_survival(t: &AnswerTriple, comp: &str) -> bool {
    answer_survives(&t.answer, comp)
}

/// Sweep `drop_rates` for one method. Non-tunable methods are evaluated once.
pub fn evaluate_method(
    method: &dyn CompressionMethod,
    triples: &[AnswerTriple],
    drop_rates: &[f64],
    survival_fn: &dyn Fn(&AnswerTriple, &str) -> bool,
) -> Vec<MethodRow> {
    let rates: Vec<f64> = if method.tunable() {
        drop_rates.to_vec()
    } else {
        vec![0.0]
    };
    let mut rows = Vec::new();
    for r in rates {
        let mut survived = 0usize;
        let mut ratios: Vec<f64> = Vec::new();
        let mut drops: Vec<f64> = Vec::new();
        for t in triples {
            let comp = method.compress(&t.text, r).unwrap_or_else(|_| t.text.clone());
            if survival_fn(t, &comp) {
                survived += 1;
            }
            ratios.push(compression_ratio(&t.text, &comp));
            drops.push(achieved_drop_rate(&t.text, &comp));
        }
        let n = triples.len();
        rows.push(MethodRow {
            method: method.name().to_string(),
            target_drop_rate: r,
            survival: if n > 0 { survived as f64 / n as f64 } else { 0.0 },
            mean_ratio: if !ratios.is_empty() {
                ratios.iter().sum::<f64>() / ratios.len() as f64
            } else {
                0.0
            },
            mean_achieved_drop: if !drops.is_empty() {
                drops.iter().sum::<f64>() / drops.len() as f64
            } else {
                0.0
            },
            n,
        });
    }
    rows
}

/// Per-triple survival booleans for one method at one drop rate.
pub fn survival_vector(
    method: &dyn CompressionMethod,
    triples: &[AnswerTriple],
    drop_rate: f64,
    survival_fn: &dyn Fn(&AnswerTriple, &str) -> bool,
) -> Vec<bool> {
    triples
        .iter()
        .map(|t| {
            let comp = method
                .compress(&t.text, drop_rate)
                .unwrap_or_else(|_| t.text.clone());
            survival_fn(t, &comp)
        })
        .collect()
}

/// Legacy paired McNemar (same-triples) returning the discordant counts and a
/// two-sided exact-binomial p. Delegates to [`mcnemar_paired`].
#[derive(Debug, Clone, Copy)]
pub struct McNemar {
    pub b01_a_worse: usize,
    pub b10_a_better: usize,
    pub n_discordant: usize,
    pub p_value: f64,
}

pub fn mcnemar(a: &[bool], b: &[bool]) -> anyhow::Result<McNemar> {
    let r = mcnemar_paired(a, b)?;
    Ok(McNemar {
        b01_a_worse: r.c,
        b10_a_better: r.b,
        n_discordant: r.n_discordant,
        p_value: r.p_exact,
    })
}

/// Result of a full benchmark run, preserving method insertion order.
pub struct BenchmarkRun {
    pub results: Vec<(String, Vec<MethodRow>)>,
    pub skipped: Vec<(String, String)>,
}

pub fn run_benchmark(
    triples: &[AnswerTriple],
    methods: &[Box<dyn CompressionMethod>],
    drop_rates: &[f64],
    survival_fn: &dyn Fn(&AnswerTriple, &str) -> bool,
) -> BenchmarkRun {
    let mut results: Vec<(String, Vec<MethodRow>)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for m in methods {
        let (ok, reason) = m.available();
        if !ok {
            skipped.push((m.name().to_string(), reason));
            continue;
        }
        results.push((
            m.name().to_string(),
            evaluate_method(m.as_ref(), triples, drop_rates, survival_fn),
        ));
    }
    BenchmarkRun { results, skipped }
}

fn ljust(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

pub fn format_report(run: &BenchmarkRun, triples: &[AnswerTriple], drop_rates: &[f64]) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("== Polymorph answer-survival benchmark ==".to_string());
    lines.push(format!(
        "triples: {}   target drop rates: {:?}",
        triples.len(),
        drop_rates
    ));
    // Fact-type breakdown (sorted).
    let mut by_type: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for t in triples {
        *by_type.entry(t.fact_type.clone()).or_insert(0) += 1;
    }
    lines.push(
        "fact types: ".to_string()
            + &by_type
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", "),
    );
    lines.push(String::new());
    lines.push("survival % (compression ratio x) at each target drop rate:".to_string());
    let header = format!(
        "  {}{}",
        ljust("method", 20),
        drop_rates
            .iter()
            .map(|r| ljust(&format!("R={r:.2}"), 12))
            .collect::<String>()
    );
    let dash = header.len().saturating_sub(2);
    lines.push(header);
    lines.push(format!("  {}", "-".repeat(dash)));
    for (name, rows) in &run.results {
        if rows.is_empty() {
            continue;
        }
        if rows.len() == 1 && rows[0].target_drop_rate == 0.0 && drop_rates.len() != 1 {
            let row = &rows[0];
            let cell = format!(
                "{:4.0}% ({:.2}x @drop{:.2})",
                row.survival * 100.0,
                row.mean_ratio,
                row.mean_achieved_drop
            );
            lines.push(format!("  {}{}   [not rate-tunable]", ljust(name, 18), cell));
            continue;
        }
        let mut cells = String::new();
        for r in drop_rates {
            match rows.iter().find(|row| row.target_drop_rate == *r) {
                None => cells.push_str(&ljust("—", 12)),
                Some(row) => cells.push_str(&ljust(
                    &format!("{:3.0}%({:.1}x)", row.survival * 100.0, row.mean_ratio),
                    12,
                )),
            }
        }
        lines.push(format!("  {}{}", ljust(name, 18), cells));
    }
    if !run.skipped.is_empty() {
        lines.push(String::new());
        lines.push("skipped methods:".to_string());
        for (name, reason) in &run.skipped {
            lines.push(format!("  {name}: {reason}"));
        }
    }
    lines.push(String::new());
    lines.push(
        "Read: higher survival at higher compression (more x / higher R) is better.\n  \
         'random' is the floor; a real ranker must keep needles the floor drops.\n  \
         survival is exact-match of the answer needle in the compressed text."
            .to_string(),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::{default_methods, DeterministicDedup, KeepSeverityHeuristic, RandomDropFloor};
    use crate::triples::curated_triples;

    const RATES: [f64; 3] = [0.2, 0.5, 0.8];

    #[test]
    fn answer_survives_normalizes_whitespace_and_case() {
        assert!(answer_survives("INC0000045", "blah   inc0000045\n more"));
        assert!(!answer_survives("INC0000045", "no needle here"));
    }

    #[test]
    fn compression_ratio_and_drop() {
        let text = "a b c d e f g h";
        let half = "a b c d";
        assert!(compression_ratio(text, half) > 1.0);
        assert!(achieved_drop_rate(text, half) > 0.0);
    }

    #[test]
    fn run_benchmark_produces_valid_rows() {
        let triples = curated_triples();
        let methods = default_methods();
        let run = run_benchmark(&triples, &methods, &RATES, &default_survival);
        let names: std::collections::BTreeSet<&str> =
            run.results.iter().map(|(n, _)| n.as_str()).collect();
        for required in ["deterministic", "keep-severity", "random"] {
            assert!(names.contains(required), "missing {required}");
        }
        for (_n, rows) in &run.results {
            for r in rows {
                assert!((0.0..=1.0).contains(&r.survival));
                assert!(r.mean_ratio >= 1.0 - 1e-9);
                assert_eq!(r.n, triples.len());
            }
        }
    }

    #[test]
    fn deterministic_preserves_all_unique_needles() {
        let triples = curated_triples();
        let rows = evaluate_method(&DeterministicDedup::default(), &triples, &RATES, &default_survival);
        assert_eq!(rows[0].survival, 1.0);
    }

    #[test]
    fn random_is_a_floor_keep_severity_beats_it_at_high_drop() {
        let triples = curated_triples();
        let sev = evaluate_method(&KeepSeverityHeuristic, &triples, &[0.8], &default_survival);
        let rnd = evaluate_method(&RandomDropFloor::default(), &triples, &[0.8], &default_survival);
        assert!(sev[0].survival >= rnd[0].survival);
    }

    #[test]
    fn format_report_is_nonempty_with_survival() {
        let triples = curated_triples();
        let methods = default_methods();
        let run = run_benchmark(&triples, &methods, &RATES, &default_survival);
        let report = format_report(&run, &triples, &RATES);
        assert!(report.contains("survival"));
    }

    #[test]
    fn mcnemar_paired_significance() {
        let a = vec![true; 100];
        let mut b = vec![true; 90];
        b.extend(vec![false; 10]);
        let r = mcnemar(&a, &b).unwrap();
        assert_eq!(r.b10_a_better, 10);
        assert_eq!(r.b01_a_worse, 0);
        assert!(r.p_value < 0.05);
        assert_eq!(mcnemar(&a, &a).unwrap().p_value, 1.0);
    }

    #[test]
    fn survival_vector_aligned_length() {
        let ts = curated_triples();
        let v = survival_vector(&KeepSeverityHeuristic, &ts, 0.5, &default_survival);
        assert_eq!(v.len(), ts.len());
    }
}
