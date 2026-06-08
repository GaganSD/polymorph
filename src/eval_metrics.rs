//! Ranking + calibrated-decode quality metrics for the LaMR pruner's per-token
//! drop-probabilities against gold tags. Pure math ported from the Python
//! `polymorph_lamr.eval.evaluate::ranking_metrics` (the torch `collect_drop_probs`
//! forward pass stays in Python; once you have `(drop_prob, gold)` arrays these
//! metrics are framework-free).

use serde::Serialize;
use serde_json::{json, Value};

/// Default fraction of tokens to drop at the calibrated-decode operating point.
pub const DEFAULT_TARGET_RATE: f64 = 0.30;

#[derive(Debug, Clone, Serialize)]
pub struct RankingMetrics {
    pub tokens: usize,
    pub gold_rate: f64,
    pub target_rate: f64,
    pub pr_auc: f64,
    pub roc_auc: f64,
    pub best_f1: f64,
    pub best_f1_thr: f64,
    pub f1_at_target: f64,
    pub prec_at_target: f64,
    pub rec_at_target: f64,
    pub accuracy_at_target: f64,
    pub thr_at_target: f64,
    pub pred_drop_rate: f64,
    pub argmax_f1: f64,
    pub argmax_prec: f64,
    pub argmax_rec: f64,
    pub argmax_drop_rate: f64,
    pub per_token_bce: f64,
}

impl RankingMetrics {
    pub fn as_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(json!({}))
    }
}

fn prf(tp: f64, fp: f64, fn_: f64) -> (f64, f64, f64) {
    let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let rec = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f1 = if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    };
    (prec, rec, f1)
}

/// Python `round` is round-half-to-even; reproduce it for the decode cut.
fn round_half_even(x: f64) -> i64 {
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        let f = x.floor() as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    } else {
        x.round() as i64
    }
}

fn empty() -> RankingMetrics {
    RankingMetrics {
        tokens: 0,
        gold_rate: 0.0,
        target_rate: 0.0,
        pr_auc: 0.0,
        roc_auc: 0.0,
        best_f1: 0.0,
        best_f1_thr: 0.0,
        f1_at_target: 0.0,
        prec_at_target: 0.0,
        rec_at_target: 0.0,
        accuracy_at_target: 0.0,
        thr_at_target: 0.0,
        pred_drop_rate: 0.0,
        argmax_f1: 0.0,
        argmax_prec: 0.0,
        argmax_rec: 0.0,
        argmax_drop_rate: 0.0,
        per_token_bce: 0.0,
    }
}

/// Threshold-free ranking quality (PR-AUC, ROC-AUC, best-F1) plus calibrated
/// decode quality at `target_rate` (the fraction of tokens to drop; `None` uses
/// the gold drop rate). `gold[i]` is 0/1. Tie order on equal drop-probs is
/// resolved by ascending index (a stable sort), matching the common case of
/// distinct continuous sigmoid outputs.
pub fn ranking_metrics(drop_prob: &[f64], gold: &[i64], target_rate: Option<f64>) -> RankingMetrics {
    let n = gold.len();
    if n == 0 {
        let mut m = empty();
        m.target_rate = target_rate.unwrap_or(0.0);
        return m;
    }
    assert_eq!(drop_prob.len(), n, "drop_prob and gold must align");
    let pos = gold.iter().filter(|&&g| g == 1).count();
    let gold_rate = pos as f64 / n as f64;
    let target_rate = target_rate.unwrap_or(gold_rate);

    // order: indices by drop_prob descending, ties by ascending index (stable).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        drop_prob[b]
            .partial_cmp(&drop_prob[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // cumulative true-positives along the descending order.
    let mut cum_tp = vec![0i64; n];
    let mut acc = 0i64;
    for (rank, &idx) in order.iter().enumerate() {
        acc += gold[idx];
        cum_tp[rank] = acc;
    }

    // PR-AUC (average precision over the drop class).
    let pr_auc = if pos > 0 {
        let mut s = 0.0;
        for (rank, &idx) in order.iter().enumerate() {
            if gold[idx] == 1 {
                s += cum_tp[rank] as f64 / (rank + 1) as f64;
            }
        }
        s / pos as f64
    } else {
        0.0
    };

    // ROC-AUC via Mann-Whitney U on ascending ranks.
    let mut asc: Vec<usize> = (0..n).collect();
    asc.sort_by(|&a, &b| {
        drop_prob[a]
            .partial_cmp(&drop_prob[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut ranks = vec![0f64; n];
    for (i, &idx) in asc.iter().enumerate() {
        ranks[idx] = (i + 1) as f64;
    }
    let neg = n - pos;
    let roc_auc = if pos > 0 && neg > 0 {
        let sum_pos: f64 = (0..n).filter(|&i| gold[i] == 1).map(|i| ranks[i]).sum();
        (sum_pos - pos as f64 * (pos as f64 + 1.0) / 2.0) / (pos as f64 * neg as f64)
    } else {
        0.0
    };

    // Best F1 across all cutoffs.
    let mut best_i = 0usize;
    let mut best_f1 = 0.0;
    for (rank, &tp) in cum_tp.iter().enumerate() {
        let denom = (rank + 1) as f64 + pos as f64;
        let f1 = if denom > 0.0 {
            2.0 * tp as f64 / denom
        } else {
            0.0
        };
        if f1 > best_f1 {
            best_f1 = f1;
            best_i = rank;
        }
    }
    let best_thr = drop_prob[order[best_i]];

    // Calibrated decode at target_rate.
    let k = (round_half_even(target_rate * n as f64).max(1) as usize).min(n);
    let tp = cum_tp[k - 1] as f64;
    let fp = k as f64 - tp;
    let fn_ = pos as f64 - tp;
    let (prec_t, rec_t, f1_t) = prf(tp, fp, fn_);
    let thr_at_target = drop_prob[order[k - 1]];
    let acc_at_target = (tp + (neg as f64 - fp)) / n as f64;

    // argmax (threshold 0.5) reference.
    let mut tp5 = 0.0;
    let mut fp5 = 0.0;
    let mut fn5 = 0.0;
    let mut pred_count = 0usize;
    for i in 0..n {
        let pred = drop_prob[i] >= 0.5;
        if pred {
            pred_count += 1;
        }
        match (pred, gold[i] == 1) {
            (true, true) => tp5 += 1.0,
            (true, false) => fp5 += 1.0,
            (false, true) => fn5 += 1.0,
            (false, false) => {}
        }
    }
    let (prec5, rec5, f15) = prf(tp5, fp5, fn5);

    // Unweighted per-token BCE.
    let bce = -(0..n)
        .map(|i| {
            let p = drop_prob[i].clamp(1e-7, 1.0 - 1e-7);
            if gold[i] == 1 {
                p.ln()
            } else {
                (1.0 - p).ln()
            }
        })
        .sum::<f64>()
        / n as f64;

    RankingMetrics {
        tokens: n,
        gold_rate,
        target_rate,
        pr_auc,
        roc_auc,
        best_f1,
        best_f1_thr: best_thr,
        f1_at_target: f1_t,
        prec_at_target: prec_t,
        rec_at_target: rec_t,
        accuracy_at_target: acc_at_target,
        thr_at_target,
        pred_drop_rate: k as f64 / n as f64,
        argmax_f1: f15,
        argmax_prec: prec5,
        argmax_rec: rec5,
        argmax_drop_rate: pred_count as f64 / n as f64,
        per_token_bce: bce,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn empty_input_is_zeroed() {
        let m = ranking_metrics(&[], &[], Some(0.3));
        assert_eq!(m.tokens, 0);
        assert_eq!(m.target_rate, 0.3);
        assert_eq!(m.pr_auc, 0.0);
    }

    #[test]
    fn hand_computed_small_example() {
        // drop_prob=[0.9,0.8,0.3,0.1], gold=[1,0,1,0], target=0.5.
        let m = ranking_metrics(&[0.9, 0.8, 0.3, 0.1], &[1, 0, 1, 0], Some(0.5));
        assert_eq!(m.tokens, 4);
        assert!(close(m.gold_rate, 0.5));
        assert!(close(m.pr_auc, (1.0 + 2.0 / 3.0) / 2.0), "pr_auc={}", m.pr_auc);
        assert!(close(m.roc_auc, 0.75), "roc_auc={}", m.roc_auc);
        assert!(close(m.best_f1, 0.8), "best_f1={}", m.best_f1);
        assert!(close(m.best_f1_thr, 0.3));
        assert!(close(m.f1_at_target, 0.5));
        assert!(close(m.prec_at_target, 0.5));
        assert!(close(m.rec_at_target, 0.5));
        assert!(close(m.thr_at_target, 0.8));
        assert!(close(m.accuracy_at_target, 0.5));
        assert!(close(m.argmax_f1, 0.5));
        assert!(close(m.argmax_drop_rate, 0.5));
        assert!(close(m.pred_drop_rate, 0.5));
        // bce = -mean[ln.9, ln.2, ln.3, ln.9] = 3.0241317/4 = 0.7560329
        assert!(close(m.per_token_bce, 0.7560329), "bce={}", m.per_token_bce);
    }

    #[test]
    fn perfect_ranking_has_unit_auc() {
        // all drops ranked strictly above all keeps -> PR-AUC = ROC-AUC = 1.
        let probs = [0.99, 0.98, 0.2, 0.1];
        let gold = [1, 1, 0, 0];
        let m = ranking_metrics(&probs, &gold, None);
        assert!(close(m.pr_auc, 1.0));
        assert!(close(m.roc_auc, 1.0));
        assert!(close(m.best_f1, 1.0));
        // target defaults to gold_rate = 0.5
        assert!(close(m.target_rate, 0.5));
    }

    #[test]
    fn all_negative_gold_is_safe() {
        let m = ranking_metrics(&[0.1, 0.2, 0.3], &[0, 0, 0], Some(0.3));
        assert_eq!(m.pr_auc, 0.0);
        assert_eq!(m.roc_auc, 0.0);
    }
}
