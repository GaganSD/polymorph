//! End-to-end text compression: turn a log into a smaller log.
//!
//! The lock/dedup/CCR modules produce *masks* and *intervals*; nothing here-to
//! assembled the actual compressed **string** a caller (the `compress_log` MCP
//! tool, a skill) wants back. This module does that, running the LaMR pruner in
//! its native **ModernBERT** token space:
//!
//! 1. **Lock** (cl100k): [`crate::lock_payload`]'s structural mask → the set of
//!    locked **byte intervals** (tokenizer-independent: AST byte ranges + DAAC
//!    keyword spans, already unioned into the mask).
//! 2. **Tokenise** the text with the ModernBERT tokenizer ([`crate::modernbert`]),
//!    the space `mb_v0` was trained in.
//! 3. **Project** the locked byte intervals onto ModernBERT tokens → a per-mb-token
//!    lock mask (an mb token overlapping any locked byte is force-kept).
//! 4. **Prune** with the ONNX model ([`crate::lamr::apply_lamr_if_model`]):
//!    windowed `P(drop)` → span-aware decode to the target rate, lock-respecting.
//! 5. **Reconstruct** by decoding the surviving mb ids — mirroring the Python
//!    `span_decode`'s `decode_tokens(kept, "modernbert")`.
//!
//! With no model configured the text is returned unchanged (never the random mock
//! — that would corrupt an audit log). Token *counts* in the result are cl100k, a
//! method-independent yardstick (matching the benchmark).

use std::path::Path;

use anyhow::Result;

use crate::{lamr, locking, modernbert, tokenizer, Language};

/// Outcome of compressing one block of text.
#[derive(Debug, Clone)]
pub struct CompressResult {
    /// The compressed text (extractive — only deletions, decoded from surviving
    /// ModernBERT tokens).
    pub compressed: String,
    /// cl100k token count of the input (method-independent yardstick).
    pub input_tokens: usize,
    /// cl100k token count of the output.
    pub output_tokens: usize,
    /// `input_tokens / max(1, output_tokens)`.
    pub ratio: f64,
    /// True iff the trained ONNX pruner actually ran (false → returned verbatim
    /// because no model is configured / loadable).
    pub used_model: bool,
}

/// Compress `text` with structural locking + the LaMR neural pruner.
///
/// `target_rate` is the fraction of unlocked tokens to drop (`None` → the model's
/// `decode.json` default, ~0.30). `max_neural_tokens` caps how many leading
/// ModernBERT tokens are scored by the model; beyond it the tail is kept verbatim
/// (a latency guard for pathological all-unique prose — a kept tail never loses a
/// needle, it just isn't pruned). `None` scores the whole sequence.
pub fn compress_text(
    text: &str,
    language: Language,
    keywords: &[String],
    grammars_dir: &Path,
    target_rate: Option<f64>,
    max_neural_tokens: Option<usize>,
) -> Result<CompressResult> {
    let input_tokens = tokenizer::count_tokens(text)?;
    if text.is_empty() {
        return Ok(CompressResult {
            compressed: String::new(),
            input_tokens,
            output_tokens: 0,
            ratio: 1.0,
            used_model: false,
        });
    }

    // 1. Structural lock (cl100k) → locked byte intervals.
    let lock = crate::compress_deterministic(text, language, keywords, grammars_dir)?;
    let locked = locked_byte_intervals(&lock.mask, &lock.token_spans);

    // 2. ModernBERT tokenisation (the model's token space).
    let mb = modernbert::get()?;
    let (mb_ids, mb_spans) = mb.encode_with_spans(text);
    if mb_ids.is_empty() {
        return Ok(CompressResult {
            compressed: text.to_string(),
            input_tokens,
            output_tokens: input_tokens,
            ratio: 1.0,
            used_model: false,
        });
    }

    // 3. Project locked byte intervals onto ModernBERT tokens.
    let mut mb_lock = project_locks(&mb_spans, &locked);

    // Latency guard: force-keep everything past the neural budget so the model
    // only scores a bounded prefix.
    let scored_len = max_neural_tokens.map_or(mb_ids.len(), |c| c.min(mb_ids.len()));
    if scored_len < mb_ids.len() {
        for l in mb_lock.iter_mut().skip(scored_len) {
            *l = true;
        }
    }

    // 4. Prune with the trained model (None → no model configured/loadable).
    let drop_mask = lamr::apply_lamr_if_model(&mb_ids, &mb_lock, &mb_spans, text, target_rate);

    let (compressed, used_model) = match drop_mask {
        Some(drop) => {
            // 5. Reconstruct from the surviving ids (ModernBERT decode).
            let kept: Vec<u32> = mb_ids
                .iter()
                .zip(drop.iter())
                .filter(|(_, &d)| !d)
                .map(|(&id, _)| id)
                .collect();
            (mb.decode(&kept)?, true)
        }
        None => (text.to_string(), false),
    };

    let output_tokens = tokenizer::count_tokens(&compressed)?;
    let ratio = input_tokens as f64 / output_tokens.max(1) as f64;
    Ok(CompressResult {
        compressed,
        input_tokens,
        output_tokens,
        ratio,
        used_model,
    })
}

/// The set of locked byte intervals: the byte span of every token the structural
/// lock marked keep, merged. Tokenizer-independent — these bytes must survive in
/// whatever token space the pruner runs in.
fn locked_byte_intervals(mask: &[bool], token_spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    debug_assert_eq!(mask.len(), token_spans.len());
    let raw: Vec<(usize, usize)> = mask
        .iter()
        .zip(token_spans.iter())
        .filter(|(&locked, _)| locked)
        .map(|(_, &span)| span)
        .collect();
    locking::sort_and_merge(raw)
}

/// Per-ModernBERT-token lock mask: token `i` is locked iff its byte span overlaps
/// any locked interval. Both inputs are sorted by start (mb spans by construction,
/// `locked` by [`locking::sort_and_merge`]), so a two-pointer sweep is O(n+m).
fn project_locks(mb_spans: &[(usize, usize)], locked: &[(usize, usize)]) -> Vec<bool> {
    let mut out = vec![false; mb_spans.len()];
    if locked.is_empty() {
        return out;
    }
    let mut j = 0usize;
    for (i, &(s, e)) in mb_spans.iter().enumerate() {
        // Advance past locked intervals that end at/before this token starts.
        while j < locked.len() && locked[j].1 <= s {
            j += 1;
        }
        // Overlap iff the next locked interval starts before this token ends.
        if j < locked.len() && locked[j].0 < e {
            out[i] = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grammars() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
    }

    #[test]
    fn locked_intervals_merge_adjacent_kept_tokens() {
        // tokens:  [keep][keep][drop][keep]   spans 0-3,3-7,7-9,9-12
        let mask = vec![true, true, false, true];
        let spans = vec![(0, 3), (3, 7), (7, 9), (9, 12)];
        let iv = locked_byte_intervals(&mask, &spans);
        assert_eq!(iv, vec![(0, 7), (9, 12)], "adjacent kept tokens merge");
    }

    #[test]
    fn project_locks_overlap_sweep() {
        // locked byte intervals [2,5) and [8,9)
        let locked = vec![(2, 5), (8, 9)];
        // mb tokens: [0,2) no, [2,4) yes, [4,7) yes(overlaps 2-5), [7,8) no, [8,10) yes
        let spans = vec![(0, 2), (2, 4), (4, 7), (7, 8), (8, 10)];
        let lk = project_locks(&spans, &locked);
        assert_eq!(lk, vec![false, true, true, false, true]);
    }

    #[test]
    fn project_locks_empty_locked_is_all_false() {
        let spans = vec![(0, 2), (2, 4)];
        assert_eq!(project_locks(&spans, &[]), vec![false, false]);
    }

    #[test]
    fn no_model_returns_text_unchanged() {
        // With no POLYMORPH_LAMR_MODEL configured, compress_text must return the
        // input verbatim (never the random mock) so an audit log is never corrupted.
        let prev = std::env::var(lamr::LAMR_MODEL_ENV).ok();
        std::env::remove_var(lamr::LAMR_MODEL_ENV);
        let text = "ERROR db connection refused at pool.rs:42\nINFO heartbeat ok\n";
        let res = compress_text(text, Language::Json, &[], &grammars(), None, None).unwrap();
        assert_eq!(res.compressed, text, "no model → verbatim");
        assert!(!res.used_model);
        assert_eq!(res.input_tokens, res.output_tokens);
        if let Some(v) = prev {
            std::env::set_var(lamr::LAMR_MODEL_ENV, v);
        }
    }

    #[test]
    fn empty_input() {
        let res = compress_text("", Language::Json, &[], &grammars(), None, None).unwrap();
        assert_eq!(res.compressed, "");
        assert_eq!(res.input_tokens, 0);
    }
}
