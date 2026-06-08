//! Subcommand dispatch for the offline bench/eval/distill tooling ported from the
//! Python `ml_pipeline` (stats, triples, loghub, label-ceiling, survival sweep,
//! and the distillation sampler filter). Each subcommand is a thin wrapper around
//! a library entry point so the Python pipeline can shell out to this binary.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use crate::methods::default_methods;
use crate::stats::Metric;
use crate::survival::{default_survival, format_report, run_benchmark};
use crate::triples::{
    build_triples_from_paths, collect_log_files, curated_triples, AnswerTriple,
};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Try to dispatch an offline subcommand. Returns `Some(result)` if `args`
/// matched a subcommand, else `None` (so the caller falls through to MCP mode).
pub fn try_run(args: &[String]) -> Option<Result<()>> {
    if has(args, "--bench-stats") {
        return Some(bench_stats(args));
    }
    if has(args, "--build-loghub") {
        return Some(build_loghub(args));
    }
    if has(args, "--label-ceiling") {
        return Some(label_ceiling(args));
    }
    if has(args, "--build-triples") {
        return Some(build_triples(args));
    }
    if has(args, "--bench-survival") {
        return Some(bench_survival(args));
    }
    if has(args, "--sampler-filter") {
        return Some(sampler_filter(args));
    }
    if has(args, "--eval-metrics") {
        return Some(eval_metrics(args));
    }
    None
}

/// Read a JSON `{"drop_prob":[...],"gold":[0|1,...],"target_rate":<f>?}` and print
/// the ranking + calibrated-decode metrics. The torch forward pass that produces
/// `drop_prob`/`gold` stays in Python; this scores them.
fn eval_metrics(args: &[String]) -> Result<()> {
    let path = flag(args, "--eval-metrics").ok_or_else(|| anyhow!("--eval-metrics needs a json path"))?;
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let drop_prob: Vec<f64> = serde_json::from_value(v["drop_prob"].clone())?;
    let gold: Vec<i64> = serde_json::from_value(v["gold"].clone())?;
    let target_rate = v.get("target_rate").and_then(|t| t.as_f64());
    let m = crate::eval_metrics::ranking_metrics(&drop_prob, &gold, target_rate);
    println!("{}", serde_json::to_string_pretty(&m.as_value())?);
    Ok(())
}

fn bench_stats(args: &[String]) -> Result<()> {
    let results = flag(args, "--bench-stats").ok_or_else(|| anyhow!("--bench-stats needs a path"))?;
    let metric = match flag(args, "--metric").unwrap_or("judge") {
        "exact" => Metric::Exact,
        _ => Metric::Judge,
    };
    let resamples = flag(args, "--resamples").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let conf = flag(args, "--conf").and_then(|s| s.parse().ok()).unwrap_or(0.95);
    let seed = flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(1234);
    let out = flag(args, "--out").map(PathBuf::from);
    crate::stats::run(Path::new(results), metric, resamples, conf, seed, out.as_deref())
}

fn build_loghub(args: &[String]) -> Result<()> {
    let raw_dir = PathBuf::from(flag(args, "--raw-dir").unwrap_or("../data/raw/loghub2"));
    let out = PathBuf::from(flag(args, "--out").unwrap_or("../data/bench/loghub_triples.json"));
    let window = flag(args, "--window-lines").and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_per = flag(args, "--max-per-system").and_then(|s| s.parse().ok()).unwrap_or(40);
    crate::loghub::run(&raw_dir, &out, window, max_per)
}

fn label_ceiling(args: &[String]) -> Result<()> {
    let distilled = flag(args, "--distilled").ok_or_else(|| anyhow!("--distilled is required"))?;
    let limit = flag(args, "--limit").and_then(|s| s.parse().ok());
    let semantic = has(args, "--semantic");
    let out = flag(args, "--out").map(PathBuf::from);
    crate::label_ceiling::run(Path::new(distilled), limit, semantic, out.as_deref())
}

fn build_triples(args: &[String]) -> Result<()> {
    let root = flag(args, "--build-triples").ok_or_else(|| anyhow!("--build-triples needs a root dir"))?;
    let window = flag(args, "--window-lines").and_then(|s| s.parse().ok()).unwrap_or(40);
    let max_per_file = flag(args, "--max-per-file").and_then(|s| s.parse().ok()).unwrap_or(5);
    let max_total = flag(args, "--max-total").and_then(|s| s.parse().ok());
    let paths = collect_log_files(Path::new(root));
    let triples = build_triples_from_paths(&paths, window, max_per_file, max_total, 2_000_000);
    let json = crate::triples::dump_json(&triples)?;
    match flag(args, "--out") {
        Some(out) => {
            std::fs::write(out, &json)?;
            eprintln!("wrote {} triples to {out}", triples.len());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn load_triples(path: &Path) -> Result<Vec<AnswerTriple>> {
    let text = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let arr = if v.is_array() {
        v
    } else {
        v.get("triples").cloned().ok_or_else(|| anyhow!("no 'triples' array in {}", path.display()))?
    };
    Ok(serde_json::from_value(arr)?)
}

fn bench_survival(args: &[String]) -> Result<()> {
    let triples = match flag(args, "--triples") {
        Some(p) => load_triples(Path::new(p))?,
        None => curated_triples(),
    };
    let rates: Vec<f64> = match flag(args, "--drop-rates") {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None => vec![0.2, 0.5, 0.8],
    };
    let methods = default_methods();
    let run = run_benchmark(&triples, &methods, &rates, &default_survival);
    println!("{}", format_report(&run, &triples, &rates));
    Ok(())
}

/// Read lines from stdin, deduplicate by training-template key and drop
/// low-signal trash, emit the surviving representative lines to stdout. Replaces
/// the per-line `normalize.template_key_cached` / `is_low_signal` loop in
/// `distill/sampler.py`.
fn sampler_filter(args: &[String]) -> Result<()> {
    let min_ratio = flag(args, "--min-ratio").and_then(|s| s.parse().ok()).unwrap_or(0.30);
    let min_alnum = flag(args, "--min-alnum").and_then(|s| s.parse().ok()).unwrap_or(24);
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut kept = 0usize;
    let mut dropped = 0usize;
    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());
    for line in input.lines() {
        // Mirror sampler.py: blank lines are skipped before dedup/gating.
        if line.trim().is_empty() {
            continue;
        }
        let key = crate::normalize::template_key(line);
        if !seen.insert(key) {
            continue;
        }
        if crate::normalize::is_low_signal(line, min_ratio, min_alnum) {
            dropped += 1;
            continue;
        }
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        kept += 1;
    }
    w.flush()?;
    eprintln!("sampler-filter: kept {kept} representatives, dropped {dropped} low-signal");
    Ok(())
}
