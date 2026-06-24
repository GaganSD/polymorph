//! Defensible-eval statistics over the saved `judge_bench` per-item JSON.
//!
//! `judge_bench` writes a `per_item` map keyed `"<method>@<ratio-key>"` whose
//! values are aligned lists of `{doc_id, fact_type, answer, ratio, exact, judge,
//! judge_error}` records (same doc_ids, same order, across every method at a given
//! ratio). This module turns that raw log into the three things a headline
//! answer-survival claim actually needs to be defensible, WITHOUT re-running the
//! (paid) judge:
//!
//!   * **Per-domain / per-fact_type survival breakdowns** — so a claim can't ride
//!     on one easy domain.
//!   * **McNemar's paired test** of two methods on the SAME triples — the χ²
//!     statistic (Yates' continuity correction) plus the exact two-sided binomial
//!     p on the discordant pairs (the χ² approximation is unreliable when the
//!     discordant count is small).
//!   * **Bootstrap 95% CIs** on each method's survival rate — a seeded,
//!     deterministic percentile bootstrap over the per-item survival booleans.
//!
//! Pure / offline (no model, no network). Ported from the Python
//! `polymorph_lamr.bench.stats`; the bootstrap uses a seeded `ChaCha8Rng` rather
//! than CPython's Mersenne Twister — internally deterministic and reproducible,
//! which is all the contract requires.

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Which survival column to read off a per-item record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Judge,
    Exact,
}

impl Metric {
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Judge => "judge",
            Metric::Exact => "exact",
        }
    }
}

/// One per-item survival record, as written by `judge_bench` into `per_item`.
#[derive(Debug, Clone, Deserialize)]
pub struct Item {
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default = "unknown_fact_type")]
    pub fact_type: String,
    #[serde(default)]
    pub judge: bool,
    #[serde(default)]
    pub exact: bool,
    #[serde(default)]
    pub judge_error: bool,
}

fn unknown_fact_type() -> String {
    "?".to_string()
}

impl Item {
    fn bit(&self, metric: Metric) -> bool {
        match metric {
            Metric::Judge => self.judge,
            Metric::Exact => self.exact,
        }
    }
}

// ---------------------------------------------------------------------------
// Domain derivation + key parsing
// ---------------------------------------------------------------------------

/// Collapse a fact_type to a coarse domain. For LogHub triples the fact_type is
/// `loghub:<domain>` (e.g. `loghub:spark`) and the domain is the suffix; for
/// semantic/structural needles (`semantic:msg`, `http_status`) the fact_type
/// already names the class, so it is returned unchanged.
pub fn domain_of(fact_type: &str) -> String {
    if let Some((head, tail)) = fact_type.split_once(':') {
        if head == "loghub" {
            return tail.to_string();
        }
    }
    fact_type.to_string()
}

/// Split a `per_item` key `"<method>@<ratio-key>"` into (method, ratio_key).
/// Method names never contain '@'; the ratio key is everything after the last
/// '@' (e.g. `lamr+span@iso3.0` -> ("lamr+span", "iso3.0")).
pub fn split_key(key: &str) -> (String, String) {
    // Mirror Python `rpartition("@")` + `if not method: return key, ""`: a key
    // with no '@', or one whose method part is empty (leading '@'), is treated as
    // an all-method key with an empty ratio.
    match key.rfind('@') {
        Some(idx) if idx > 0 => (key[..idx].to_string(), key[idx + 1..].to_string()),
        _ => (key.to_string(), String::new()),
    }
}

/// Distinct ratio keys present, preserving first-seen order.
pub fn ratio_keys(per_item: &BTreeMap<String, Vec<Item>>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for k in per_item.keys() {
        let r = split_key(k).1;
        if !seen.contains(&r) {
            seen.push(r);
        }
    }
    seen
}

/// Method names available at a given ratio key, in first-seen order.
pub fn methods_at(per_item: &BTreeMap<String, Vec<Item>>, ratio_key: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in per_item.keys() {
        let (m, r) = split_key(k);
        if r == ratio_key && !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Survival vectors + rates
// ---------------------------------------------------------------------------

/// Per-item survival booleans for one metric.
pub fn survival_bits(items: &[Item], metric: Metric) -> Vec<bool> {
    items.iter().map(|it| it.bit(metric)).collect()
}

/// Fraction of items that survived under `metric`.
pub fn survival_rate(items: &[Item], metric: Metric) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let survived = items.iter().filter(|it| it.bit(metric)).count();
    survived as f64 / items.len() as f64
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

// ---------------------------------------------------------------------------
// Bootstrap CI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BootstrapCI {
    pub point: f64,
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub resamples: usize,
    pub conf: f64,
}

impl BootstrapCI {
    pub fn as_dict(&self) -> Value {
        json!({
            "point": round4(self.point),
            "lo": round4(self.lo),
            "hi": round4(self.hi),
            "n": self.n,
            "resamples": self.resamples,
            "conf": self.conf,
        })
    }
}

/// Percentile bootstrap CI on a survival rate (mean of booleans).
///
/// Deterministic given (bits, resamples, conf, seed): a private `ChaCha8Rng` is
/// seeded so the same inputs always yield the same interval. Each resample draws
/// `n` items with replacement and records the resample mean; the CI is the
/// empirical [alpha/2, 1-alpha/2] percentile of those means.
pub fn bootstrap_ci(bits: &[bool], resamples: usize, conf: f64, seed: u64) -> BootstrapCI {
    let n = bits.len();
    if n == 0 {
        return BootstrapCI {
            point: 0.0,
            lo: 0.0,
            hi: 0.0,
            n: 0,
            resamples,
            conf,
        };
    }
    let survived = bits.iter().filter(|b| **b).count();
    let point = survived as f64 / n as f64;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vals: Vec<f64> = bits.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
    let mut means: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut s = 0.0;
        for _ in 0..n {
            s += vals[rng.gen_range(0..n)];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - conf;
    let lo_idx = (((alpha / 2.0) * resamples as f64) as usize).min(resamples - 1);
    let hi_idx = (((1.0 - alpha / 2.0) * resamples as f64) as usize).min(resamples - 1);
    BootstrapCI {
        point,
        lo: means[lo_idx],
        hi: means[hi_idx],
        n,
        resamples,
        conf,
    }
}

// ---------------------------------------------------------------------------
// McNemar's paired test
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct McNemarResult {
    pub b: usize, // A right / B wrong  (A better)
    pub c: usize, // A wrong / B right  (A worse)
    pub n_discordant: usize,
    pub chi2: f64,      // Yates-corrected chi-square (1 dof)
    pub p_chi2: f64,    // p from the chi-square approximation
    pub p_exact: f64,   // two-sided exact binomial p on the discordant pairs
    pub a_better: bool, // b > c
}

impl McNemarResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "b_a_better": self.b,
            "c_a_worse": self.c,
            "n_discordant": self.n_discordant,
            "chi2_yates": round4(self.chi2),
            "p_chi2": round6(self.p_chi2),
            "p_exact": round6(self.p_exact),
            "a_better": self.a_better,
        })
    }
}

/// Survival function P(X > x) for a chi-square with 1 dof, = erfc(sqrt(x/2)).
fn chi2_sf_1dof(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    libm::erfc((x / 2.0).sqrt())
}

/// Two-sided exact binomial tail at theta=0.5 over `n` discordant pairs, with
/// `k = min(b, c)`. Computes `sum_{i=0}^{k} C(n,i) * 0.5^n` incrementally to
/// avoid the huge intermediates of an explicit `comb(n,i)`.
fn exact_binomial_two_sided(n: usize, k: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    // term_0 = C(n,0) * 0.5^n = 0.5^n; term_i = term_{i-1} * (n-i+1)/i.
    let mut term = 0.5_f64.powi(n as i32);
    let mut tail = term;
    for i in 1..=k {
        term *= (n - i + 1) as f64 / i as f64;
        tail += term;
    }
    (2.0 * tail).min(1.0)
}

/// McNemar's test on aligned survival vectors of methods A and B.
///
/// `b` = A survived / B did not (A better); `c` = A did not / B did (A worse).
/// χ² uses Yates' continuity correction: (|b - c| - 1)^2 / (b + c). The exact
/// two-sided binomial p (theta = 0.5 on the discordant pairs) is the robust
/// fallback for small discordant counts. Errors if the vectors are misaligned.
pub fn mcnemar_paired(a_bits: &[bool], b_bits: &[bool]) -> Result<McNemarResult> {
    if a_bits.len() != b_bits.len() {
        return Err(anyhow!(
            "survival vectors must be aligned (same triples/order)"
        ));
    }
    let mut b = 0usize;
    let mut c = 0usize;
    for (x, y) in a_bits.iter().zip(b_bits.iter()) {
        if *x && !*y {
            b += 1;
        } else if !*x && *y {
            c += 1;
        }
    }
    let n = b + c;
    if n == 0 {
        return Ok(McNemarResult {
            b: 0,
            c: 0,
            n_discordant: 0,
            chi2: 0.0,
            p_chi2: 1.0,
            p_exact: 1.0,
            a_better: false,
        });
    }
    let diff = (b as f64 - c as f64).abs();
    let chi2 = (diff - 1.0).powi(2) / n as f64;
    let p_chi2 = chi2_sf_1dof(chi2);
    let k = b.min(c);
    let p_exact = exact_binomial_two_sided(n, k);
    Ok(McNemarResult {
        b,
        c,
        n_discordant: n,
        chi2,
        p_chi2,
        p_exact,
        a_better: b > c,
    })
}

// ---------------------------------------------------------------------------
// Aggregate analysis over a per_item map
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MethodStats {
    pub method: String,
    pub ratio_key: String,
    pub n: usize,
    pub judge: BootstrapCI,
    pub exact: BootstrapCI,
    pub judge_errors: usize,
    pub per_domain: Value,
}

impl MethodStats {
    pub fn as_dict(&self) -> Value {
        json!({
            "method": self.method,
            "ratio_key": self.ratio_key,
            "n": self.n,
            "judge": self.judge.as_dict(),
            "exact": self.exact.as_dict(),
            "judge_errors": self.judge_errors,
            "per_domain": self.per_domain,
        })
    }
}

fn per_domain(items: &[Item]) -> Value {
    let mut buckets: BTreeMap<String, Vec<&Item>> = BTreeMap::new();
    for it in items {
        buckets
            .entry(domain_of(&it.fact_type))
            .or_default()
            .push(it);
    }
    let mut out = serde_json::Map::new();
    for (d, b) in buckets {
        let judge_survived = b.iter().filter(|x| x.judge).count();
        let exact_survived = b.iter().filter(|x| x.exact).count();
        let n = b.len();
        let judge_rate = if n > 0 {
            judge_survived as f64 / n as f64
        } else {
            0.0
        };
        let exact_rate = if n > 0 {
            exact_survived as f64 / n as f64
        } else {
            0.0
        };
        out.insert(
            d,
            json!({
                "n": n,
                "judge_rate": round4(judge_rate),
                "judge_survived": judge_survived,
                "exact_rate": round4(exact_rate),
                "exact_survived": exact_survived,
            }),
        );
    }
    Value::Object(out)
}

pub fn method_stats(
    items: &[Item],
    method: &str,
    ratio_key: &str,
    resamples: usize,
    conf: f64,
    seed: u64,
) -> MethodStats {
    MethodStats {
        method: method.to_string(),
        ratio_key: ratio_key.to_string(),
        n: items.len(),
        judge: bootstrap_ci(&survival_bits(items, Metric::Judge), resamples, conf, seed),
        exact: bootstrap_ci(&survival_bits(items, Metric::Exact), resamples, conf, seed),
        judge_errors: items.iter().filter(|x| x.judge_error).count(),
        per_domain: per_domain(items),
    }
}

#[derive(Debug, Clone)]
pub struct PairTest {
    pub ratio_key: String,
    pub method_a: String,
    pub method_b: String,
    pub metric: Metric,
    pub result: McNemarResult,
}

impl PairTest {
    pub fn as_dict(&self) -> Value {
        let mut v = self.result.as_dict();
        let obj = v.as_object_mut().unwrap();
        obj.insert("ratio_key".into(), json!(self.ratio_key));
        obj.insert("method_a".into(), json!(self.method_a));
        obj.insert("method_b".into(), json!(self.method_b));
        obj.insert("metric".into(), json!(self.metric.as_str()));
        v
    }
}

/// Default paired comparisons a survival claim rests on: the model vs the
/// heuristic floor and vs the published external baseline.
pub fn default_pairs() -> Vec<(String, String)> {
    vec![
        ("lamr+span".into(), "keep-severity".into()),
        ("lamr+span".into(), "llmlingua2".into()),
        ("lamr+span+floor".into(), "keep-severity".into()),
    ]
}

/// Full defensible-eval analysis over a per_item map. Returns
/// `{ratio_key: {"methods": [...], "pairs": [...]}}`. Paired tests require the
/// two methods to be aligned on doc_id at that ratio (errors otherwise).
pub fn analyze(
    per_item: &BTreeMap<String, Vec<Item>>,
    pairs: &[(String, String)],
    metrics: &[Metric],
    resamples: usize,
    conf: f64,
    seed: u64,
) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for rk in ratio_keys(per_item) {
        let mut ms: Vec<MethodStats> = Vec::new();
        let mut method_items: BTreeMap<String, &Vec<Item>> = BTreeMap::new();
        for m in methods_at(per_item, &rk) {
            let key = format!("{m}@{rk}");
            let items = &per_item[&key];
            method_items.insert(m.clone(), items);
            ms.push(method_stats(items, &m, &rk, resamples, conf, seed));
        }
        let mut pts: Vec<PairTest> = Vec::new();
        for (a, b) in pairs {
            let (ia, ib) = match (method_items.get(a), method_items.get(b)) {
                (Some(ia), Some(ib)) => (*ia, *ib),
                _ => continue,
            };
            let ida: Vec<&Option<String>> = ia.iter().map(|x| &x.doc_id).collect();
            let idb: Vec<&Option<String>> = ib.iter().map(|x| &x.doc_id).collect();
            if ida != idb {
                return Err(anyhow!(
                    "paired methods {a} vs {b} @ {rk} are not aligned on doc_id \
                     (McNemar requires identical triples in the same order)"
                ));
            }
            for metric in metrics {
                let res = mcnemar_paired(&survival_bits(ia, *metric), &survival_bits(ib, *metric))?;
                pts.push(PairTest {
                    ratio_key: rk.clone(),
                    method_a: a.clone(),
                    method_b: b.clone(),
                    metric: *metric,
                    result: res,
                });
            }
        }
        out.insert(
            rk.clone(),
            json!({
                "methods": ms.iter().map(|m| m.as_dict()).collect::<Vec<_>>(),
                "pairs": pts.iter().map(|p| p.as_dict()).collect::<Vec<_>>(),
            }),
        );
    }
    Ok(Value::Object(out))
}

/// Load the `per_item` map from a judge_bench results JSON.
pub fn load_per_item(path: &Path) -> Result<BTreeMap<String, Vec<Item>>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let payload: Value = serde_json::from_str(&text)?;
    let pi = payload.get("per_item").ok_or_else(|| {
        anyhow!(
            "{}: no 'per_item' map (not a judge_bench results file?)",
            path.display()
        )
    })?;
    let map: BTreeMap<String, Vec<Item>> = serde_json::from_value(pi.clone())?;
    Ok(map)
}

pub fn analyze_file(path: &Path, resamples: usize, conf: f64, seed: u64) -> Result<Value> {
    let per_item = load_per_item(path)?;
    analyze(
        &per_item,
        &default_pairs(),
        &[Metric::Judge, Metric::Exact],
        resamples,
        conf,
        seed,
    )
}

// ---------------------------------------------------------------------------
// Pretty report
// ---------------------------------------------------------------------------

fn ci_cell(ci: &Value) -> String {
    let point = ci["point"].as_f64().unwrap_or(0.0);
    let lo = ci["lo"].as_f64().unwrap_or(0.0);
    let hi = ci["hi"].as_f64().unwrap_or(0.0);
    format!(
        "{:4.0}% [{:.0}-{:.0}]",
        100.0 * point,
        100.0 * lo,
        100.0 * hi
    )
}

fn ljust(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

pub fn format_stats(analysis: &Value, metric: Metric) -> String {
    let mkey = metric.as_str();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "== Defensible answer-survival stats (metric={mkey}, 95% bootstrap CI) =="
    ));
    let obj = match analysis.as_object() {
        Some(o) => o,
        None => return lines.join("\n"),
    };
    for (rk, blob) in obj {
        lines.push(String::new());
        lines.push(format!("-- ratio={rk} --"));
        let hdr = format!(
            "  {}{}{}err",
            ljust("method", 20),
            ljust("n", 5),
            ljust(&format!("{mkey} survival (95% CI)"), 22)
        );
        let dash_len = hdr.len().saturating_sub(2);
        lines.push(hdr);
        lines.push(format!("  {}", "-".repeat(dash_len)));
        let methods = blob["methods"].as_array().cloned().unwrap_or_default();
        for m in &methods {
            let ci = &m[mkey];
            lines.push(format!(
                "  {}{}{}{}",
                ljust(m["method"].as_str().unwrap_or(""), 20),
                ljust(&m["n"].to_string(), 5),
                ljust(&ci_cell(ci), 22),
                m["judge_errors"]
            ));
        }
        // per-domain for lamr+span if present, else first method.
        let focus = methods
            .iter()
            .find(|m| m["method"].as_str() == Some("lamr+span"))
            .or_else(|| methods.first());
        if let Some(focus) = focus {
            lines.push(format!(
                "  per-domain ({}, {mkey}):",
                focus["method"].as_str().unwrap_or("")
            ));
            let rate_key = format!("{mkey}_rate");
            let surv_key = format!("{mkey}_survived");
            if let Some(pd) = focus["per_domain"].as_object() {
                for (d, dd) in pd {
                    lines.push(format!(
                        "    {} n={} {:3.0}%  ({}/{})",
                        ljust(d, 14),
                        ljust(&dd["n"].to_string(), 4),
                        100.0 * dd[&rate_key].as_f64().unwrap_or(0.0),
                        dd[&surv_key],
                        dd["n"]
                    ));
                }
            }
        }
        // paired tests for this metric.
        let pairs = blob["pairs"].as_array().cloned().unwrap_or_default();
        let pts: Vec<&Value> = pairs
            .iter()
            .filter(|p| p["metric"].as_str() == Some(mkey))
            .collect();
        if !pts.is_empty() {
            lines.push("  McNemar (paired, same triples):".to_string());
            for p in pts {
                let p_exact = p["p_exact"].as_f64().unwrap_or(1.0);
                let a_better = p["a_better"].as_bool().unwrap_or(false);
                let sig = if p_exact < 0.05 && a_better {
                    "  [A significantly better]"
                } else {
                    ""
                };
                lines.push(format!(
                    "    {} vs {}: b(A>B)={} c(A<B)={} chi2={:.2} p_chi2={:.4} p_exact={:.4}{}",
                    p["method_a"].as_str().unwrap_or(""),
                    p["method_b"].as_str().unwrap_or(""),
                    p["b_a_better"],
                    p["c_a_worse"],
                    p["chi2_yates"].as_f64().unwrap_or(0.0),
                    p["p_chi2"].as_f64().unwrap_or(0.0),
                    p_exact,
                    sig,
                ));
            }
        }
    }
    lines.join("\n")
}

/// CLI entry: `polymorph-mcp --bench-stats <results.json> [--metric judge|exact]
/// [--resamples N] [--conf C] [--seed S] [--out path]`.
pub fn run(
    results: &Path,
    metric: Metric,
    resamples: usize,
    conf: f64,
    seed: u64,
    out: Option<&Path>,
) -> Result<()> {
    let analysis = analyze_file(results, resamples, conf, seed)?;
    println!("{}", format_stats(&analysis, metric));
    if let Some(out) = out {
        std::fs::write(out, serde_json::to_string_pretty(&analysis)?)?;
        println!("\nwrote {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_items(facts: &[&str], judges: &[bool]) -> Vec<Item> {
        facts
            .iter()
            .zip(judges.iter())
            .enumerate()
            .map(|(i, (ft, j))| Item {
                doc_id: Some(format!("d{i}")),
                fact_type: ft.to_string(),
                judge: *j,
                exact: *j,
                judge_error: false,
            })
            .collect()
    }

    // ---- domain / key helpers ----

    #[test]
    fn domain_of_loghub_and_passthrough() {
        assert_eq!(domain_of("loghub:spark"), "spark");
        assert_eq!(domain_of("loghub:bgl"), "bgl");
        assert_eq!(domain_of("semantic:msg"), "semantic:msg");
        assert_eq!(domain_of("http_status"), "http_status");
    }

    #[test]
    fn split_key_cases() {
        assert_eq!(
            split_key("lamr+span@iso3.0"),
            ("lamr+span".to_string(), "iso3.0".to_string())
        );
        assert_eq!(
            split_key("keep-severity@0.5"),
            ("keep-severity".to_string(), "0.5".to_string())
        );
        assert_eq!(
            split_key("noatsign"),
            ("noatsign".to_string(), String::new())
        );
        // leading '@' (empty method) is treated as an all-method key (mirrors
        // Python rpartition + `if not method`).
        assert_eq!(split_key("@b"), ("@b".to_string(), String::new()));
        assert_eq!(split_key("a@b@c"), ("a@b".to_string(), "c".to_string()));
    }

    // ---- McNemar on a known 2x2 ----

    #[test]
    fn mcnemar_known_2x2() {
        let a = vec![true; 100];
        let mut b = vec![true; 90];
        b.extend(vec![false; 10]);
        let r = mcnemar_paired(&a, &b).unwrap();
        assert_eq!(r.b, 10);
        assert_eq!(r.c, 0);
        assert_eq!(r.n_discordant, 10);
        assert!(r.a_better);
        assert!((r.chi2 - 8.1).abs() < 1e-9);
        assert!((r.p_exact - 2.0 / 1024.0).abs() < 1e-12);
        assert!(r.p_exact < 0.05);
        assert!(r.p_chi2 < 0.05);
    }

    #[test]
    fn mcnemar_no_discordance_is_p1() {
        let a = vec![true, false, true, false];
        let r = mcnemar_paired(&a, &a).unwrap();
        assert_eq!(r.n_discordant, 0);
        assert_eq!(r.p_exact, 1.0);
        assert_eq!(r.p_chi2, 1.0);
        assert!(!r.a_better);
    }

    #[test]
    fn mcnemar_symmetric_is_not_significant() {
        let mut a = vec![true; 5];
        a.extend(vec![false; 5]);
        let mut b = vec![false; 5];
        b.extend(vec![true; 5]);
        let r = mcnemar_paired(&a, &b).unwrap();
        assert_eq!(r.b, 5);
        assert_eq!(r.c, 5);
        assert!((r.p_exact - 1.0).abs() < 1e-9);
        assert!(!r.a_better);
    }

    #[test]
    fn mcnemar_rejects_misaligned() {
        assert!(mcnemar_paired(&[true, false], &[true]).is_err());
    }

    // ---- bootstrap determinism + sanity ----

    #[test]
    fn bootstrap_is_deterministic_for_fixed_seed() {
        let mut bits = vec![true; 30];
        bits.extend(vec![false; 20]);
        let a = bootstrap_ci(&bits, 500, 0.95, 7);
        let b = bootstrap_ci(&bits, 500, 0.95, 7);
        assert_eq!(a, b);
        assert!((a.point - 0.6).abs() < 1e-9);
        assert!(0.0 <= a.lo && a.lo <= a.point && a.point <= a.hi && a.hi <= 1.0);
    }

    #[test]
    fn bootstrap_different_seed_brackets_point() {
        let mut bits = vec![true; 30];
        bits.extend(vec![false; 20]);
        let a = bootstrap_ci(&bits, 500, 0.95, 1);
        let b = bootstrap_ci(&bits, 500, 0.95, 2);
        assert!((a.point - 0.6).abs() < 1e-9);
        assert!((b.point - 0.6).abs() < 1e-9);
        assert!(a.lo <= 0.6 && 0.6 <= a.hi);
        assert!(b.lo <= 0.6 && 0.6 <= b.hi);
    }

    #[test]
    fn bootstrap_degenerate_all_true_is_tight() {
        let ci = bootstrap_ci(&vec![true; 40], 300, 0.95, 3);
        assert_eq!(ci.point, 1.0);
        assert_eq!(ci.lo, 1.0);
        assert_eq!(ci.hi, 1.0);
    }

    #[test]
    fn bootstrap_empty() {
        let ci = bootstrap_ci(&[], 100, 0.95, 3);
        assert_eq!(ci.n, 0);
        assert_eq!(ci.point, 0.0);
    }

    #[test]
    fn bootstrap_ci_widens_with_smaller_n() {
        let mut small_bits = vec![true; 6];
        small_bits.extend(vec![false; 4]);
        let mut large_bits = vec![true; 60];
        large_bits.extend(vec![false; 40]);
        let small = bootstrap_ci(&small_bits, 1000, 0.95, 11);
        let large = bootstrap_ci(&large_bits, 1000, 0.95, 11);
        assert!((small.hi - small.lo) >= (large.hi - large.lo));
    }

    // ---- survival rate helpers ----

    #[test]
    fn survival_rate_and_bits() {
        let items = mk_items(&["x", "y"], &[true, false]);
        // exact mirrors judge in mk_items; build a custom case for exact != judge.
        let items = vec![
            Item {
                doc_id: None,
                fact_type: "x".into(),
                judge: true,
                exact: false,
                judge_error: false,
            },
            Item {
                doc_id: None,
                fact_type: "y".into(),
                judge: false,
                exact: true,
                judge_error: false,
            },
            items[0].clone(),
        ];
        let bits = survival_bits(&items[..2], Metric::Judge);
        assert_eq!(bits, vec![true, false]);
        assert!((survival_rate(&items[..2], Metric::Judge) - 0.5).abs() < 1e-9);
        assert!((survival_rate(&items[..2], Metric::Exact) - 0.5).abs() < 1e-9);
    }

    // ---- per-domain via method_stats ----

    #[test]
    fn method_stats_per_domain_breakdown() {
        let items = vec![
            Item {
                doc_id: Some("a".into()),
                fact_type: "loghub:spark".into(),
                judge: true,
                exact: true,
                judge_error: false,
            },
            Item {
                doc_id: Some("b".into()),
                fact_type: "loghub:spark".into(),
                judge: false,
                exact: false,
                judge_error: false,
            },
            Item {
                doc_id: Some("c".into()),
                fact_type: "loghub:bgl".into(),
                judge: true,
                exact: false,
                judge_error: false,
            },
        ];
        let ms = method_stats(&items, "lamr+span", "iso3.0", 200, 0.95, 5);
        assert_eq!(ms.n, 3);
        let pd = ms.per_domain.as_object().unwrap();
        let keys: std::collections::BTreeSet<&String> = pd.keys().collect();
        let expected: std::collections::BTreeSet<String> = ["spark".to_string(), "bgl".to_string()]
            .into_iter()
            .collect();
        assert_eq!(keys, expected.iter().collect());
        assert_eq!(pd["spark"]["n"], 2);
        assert_eq!(pd["spark"]["judge_survived"], 1);
        assert!((pd["spark"]["judge_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(pd["bgl"]["n"], 1);
        assert!((pd["bgl"]["judge_rate"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    // ---- end-to-end analyze over a paired per_item map ----

    #[test]
    fn analyze_end_to_end_pairs_and_cis() {
        let facts = ["loghub:spark", "loghub:spark", "loghub:bgl", "loghub:bgl"];
        let mut per_item: BTreeMap<String, Vec<Item>> = BTreeMap::new();
        per_item.insert(
            "lamr+span@iso3.0".into(),
            mk_items(&facts, &[true, true, true, false]),
        );
        per_item.insert(
            "keep-severity@iso3.0".into(),
            mk_items(&facts, &[false, true, false, false]),
        );
        let res = analyze(
            &per_item,
            &default_pairs(),
            &[Metric::Judge, Metric::Exact],
            200,
            0.95,
            9,
        )
        .unwrap();
        let blob = &res["iso3.0"];
        let names: std::collections::BTreeSet<&str> = blob["methods"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["method"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["keep-severity", "lamr+span"].into_iter().collect());
        let pair: Vec<&Value> = blob["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| {
                p["method_a"].as_str() == Some("lamr+span") && p["metric"].as_str() == Some("judge")
            })
            .collect();
        assert_eq!(pair.len(), 1);
        let p = pair[0];
        assert_eq!(p["b_a_better"], 2);
        assert_eq!(p["c_a_worse"], 0);
        assert_eq!(p["a_better"], true);
        for m in blob["methods"].as_array().unwrap() {
            assert!(m["judge"].get("lo").is_some());
            assert!(m["judge"].get("hi").is_some());
        }
    }

    #[test]
    fn analyze_raises_on_misaligned_pairs() {
        let mut per_item: BTreeMap<String, Vec<Item>> = BTreeMap::new();
        per_item.insert(
            "lamr+span@iso3.0".into(),
            vec![Item {
                doc_id: Some("x".into()),
                fact_type: "loghub:spark".into(),
                judge: true,
                exact: true,
                judge_error: false,
            }],
        );
        per_item.insert(
            "keep-severity@iso3.0".into(),
            vec![Item {
                doc_id: Some("y".into()),
                fact_type: "loghub:spark".into(),
                judge: false,
                exact: false,
                judge_error: false,
            }],
        );
        assert!(analyze(
            &per_item,
            &default_pairs(),
            &[Metric::Judge, Metric::Exact],
            100,
            0.95,
            1
        )
        .is_err());
    }
}
