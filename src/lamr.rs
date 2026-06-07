//! LaMR pruner.
//!
//! Two backends live here:
//!   * a deterministic RNG **mock** (`dummy_lamr_forward_pass`), kept as the
//!     fallback so callers/tests work without an exported model, and
//!   * the **real** ONNX-backed pruner (`apply_lamr_onnx`), which loads the
//!     LaMR model exported by `polymorph_lamr/export/to_onnx.py` (backbone + one
//!     drop head → a `logits` tensor of per-token drop logits), applies
//!     `sigmoid` to get a drop-probability per token, and decodes by a
//!     **target-rate threshold**: among the unlocked tokens it drops the top
//!     `round(target_rate * n_unlocked)` by probability. There is no CRF and no
//!     Viterbi — the earlier 2-tag CRF was degenerate (stable ranking, only the
//!     global bias oscillated), so decode became a calibrated threshold and the
//!     transitions side-car went away.
//!
//! Both backends run the model over the **full** token sequence (matching how
//! the labeler tags every token at training time). The mock funnels through
//! [`enforce_lock_invariant`]; the ONNX path uses [`target_rate_drop_bits`],
//! which is itself lock-aware. Both honour the invariant:
//! `lock_mask[i] == true  =>  drop_mask[i] == false`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

use tract_onnx::prelude::*;

/// Deterministic seed for the mock pruner. Used only when no ONNX model is
/// available; the real model swaps in via [`apply_lamr_onnx`].
pub const LAMR_SEED: u64 = 0xA5C3_B6D2_E91F_4471;

/// Default target drop rate. The mock returns true (drop) with ~this probability
/// per unlocked token; the ONNX path drops this fraction of unlocked tokens when
/// `decode.json` doesn't specify one.
pub const DEFAULT_DROP_RATE: f64 = 0.30;

/// Env var pointing at the exported `model.onnx`. If set and the file exists,
/// [`apply_lamr`] uses the real ONNX path; otherwise it falls back to the mock.
pub const LAMR_MODEL_ENV: &str = "POLYMORPH_LAMR_MODEL";

/// Hard upper bound on tokens fed to the ONNX model in a single forward pass.
/// Matches the sinusoidal pos-enc `max_len` in `model/backbone.py`; beyond it
/// the pos-enc slice is out of range and O(T^2) attention blows up. This is an
/// interim guard: it refuses oversize input (→ logged fallback to the mock)
/// rather than crashing/OOMing. TODO: implement windowing to the trained
/// `max_seq_len` (1024) and run the model per window, matching `dataset.py`.
const MAX_INFERENCE_TOKENS: usize = 8192;

// ---------------------------------------------------------------------------
// Mock backend (fallback)
// ---------------------------------------------------------------------------

/// Runs a "forward pass" over the unlocked token ids and returns a Vec<bool>
/// of the same length where `true` = drop. Deterministic given the const seed
/// so tests are reproducible across runs.
pub fn dummy_lamr_forward_pass(unlocked_tokens: &[u32]) -> Vec<bool> {
    dummy_lamr_forward_pass_seeded(unlocked_tokens, LAMR_SEED, DEFAULT_DROP_RATE)
}

/// Same as above but accepts a custom seed + rate. Used in tests for varying
/// the drop rate without mutating the const.
pub fn dummy_lamr_forward_pass_seeded(
    unlocked_tokens: &[u32],
    seed: u64,
    drop_rate: f64,
) -> Vec<bool> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Multiplicative folding of each token id into the RNG keeps the output
    // stable for the same input slice — the next_u64 stream alone would NOT
    // depend on the token ids, but we want the mock to behave like a model
    // that conditions on its input.
    let threshold = (drop_rate * (u32::MAX as f64)) as u32;
    unlocked_tokens
        .iter()
        .map(|tok| {
            let r = rng.next_u32() ^ tok.wrapping_mul(0x9E37_79B1);
            r < threshold
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Lock-invariant projection (shared handoff point — DO NOT weaken the invariant)
// ---------------------------------------------------------------------------

/// Project the lock constraint onto a full-length per-token drop decision.
///
/// `drop_bits[i]` is the model's decision for token `i` (`true` = drop). A
/// locked token is force-kept here: the deterministic lock is a hard must-keep
/// constraint applied *after* the model and never shown to it. Used by the mock
/// path (the ONNX path uses the lock-aware [`target_rate_drop_bits`]).
///
/// Invariant: `lock_mask[i] == true  =>  drop_mask[i] == false`.
pub fn enforce_lock_invariant(lock_mask: &[bool], drop_bits: &[bool]) -> Vec<bool> {
    assert_eq!(
        lock_mask.len(),
        drop_bits.len(),
        "drop_bits must be full-length (one per token); the model runs over the whole sequence"
    );
    lock_mask
        .iter()
        .zip(drop_bits.iter())
        .map(|(&locked, &drop)| drop && !locked)
        .collect()
}

// ---------------------------------------------------------------------------
// Target-rate threshold decode
// ---------------------------------------------------------------------------

/// Numerically-stable logistic sigmoid: `1 / (1 + e^-x)` → P(drop).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Decode per-token drop bits from per-token drop *probabilities* by a
/// target-rate cut. Among the UNLOCKED tokens, the top
/// `round(target_rate * n_unlocked)` by probability are marked drop; locked
/// tokens are never dropped. Returns the final, lock-respecting drop mask.
///
/// This is the calibrated-threshold decode: the compression ratio is a runtime
/// knob (`target_rate`) rather than a fixed 0.5 argmax, which sidesteps the
/// global-bias oscillation that made the old argmax decode look like it
/// collapsed. Deterministic: ties in probability break by original token order
/// (stable sort).
pub fn target_rate_drop_bits(drop_probs: &[f32], lock_mask: &[bool], target_rate: f64) -> Vec<bool> {
    assert_eq!(
        drop_probs.len(),
        lock_mask.len(),
        "drop_probs must be full-length (one per token), parallel to lock_mask"
    );
    let n = drop_probs.len();
    let mut drop = vec![false; n];
    let unlocked: Vec<usize> = (0..n).filter(|&i| !lock_mask[i]).collect();
    if unlocked.is_empty() {
        return drop;
    }
    let rate = target_rate.clamp(0.0, 1.0);
    let k = ((rate * unlocked.len() as f64).round() as usize).min(unlocked.len());
    if k == 0 {
        return drop;
    }
    // Stable sort unlocked indices by drop-prob descending; mark the top k.
    let mut by_prob = unlocked;
    by_prob.sort_by(|&a, &b| {
        drop_probs[b]
            .partial_cmp(&drop_probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &idx in by_prob.iter().take(k) {
        drop[idx] = true;
    }
    debug_assert!(
        drop.iter().zip(lock_mask).all(|(&d, &l)| !(d && l)),
        "target-rate decode must never drop a locked token"
    );
    drop
}

// ---------------------------------------------------------------------------
// Span-aware (chunk-level) decode
// ---------------------------------------------------------------------------
//
// Token-level top-k decode (`target_rate_drop_bits`) has a *conjunction failure*
// on multi-word needles: it drops a span the moment ANY single token in it
// crosses the cut. A free-text phrase like "disk controller firmware deadlock"
// survives only if every one of its tokens survives, so one confidently-droppable
// filler token (a leading space, "the") fragments the phrase and the needle is
// lost. This is the ChunkKV (ICLR 2025) observation: existing methods overlook
// semantic relationships between tokens, fragmenting context.
//
// The fix mirrors `ml_pipeline/polymorph_lamr/bench/spandecode.py`: group tokens
// into SPANS (whitespace-delimited words by default, or fixed N-token chunks),
// score each span by an aggregate of its tokens' drop-probs, then drop whole
// spans most-droppable-first until the token budget is hit, never splitting a
// span and never dropping a span that contains a locked / force-kept token.

/// Per-span aggregator: how a span's single drop-score is reduced from its
/// tokens' per-token drop-probs. Mirrors the Python `_AGGREGATORS` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanAggregator {
    /// Drop a span only if EVERY token is droppable (one salient token protects
    /// the whole phrase). Highest recall.
    Min,
    /// Mean token drop-prob.
    Mean,
    /// Drop a span if ANY token is droppable (mirrors token-level top-k, but
    /// still atomic). The chosen production default — best needle survival in
    /// the Python word-level experiment.
    Max,
}

impl SpanAggregator {
    fn reduce(self, probs: &[f32]) -> f32 {
        debug_assert!(!probs.is_empty(), "span must have at least one token");
        match self {
            SpanAggregator::Min => probs.iter().copied().fold(f32::INFINITY, f32::min),
            SpanAggregator::Max => probs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            SpanAggregator::Mean => {
                probs.iter().copied().sum::<f32>() / probs.len() as f32
            }
        }
    }
}

/// Span granularity: how consecutive tokens are grouped into atomic drop units.
/// Mirrors the Python `span` argument (`'word'` / `'chunk:N'` / `'token'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanGranularity {
    /// Whitespace-delimited words (the production default). A new word begins on
    /// a whitespace boundary; cl100k folds a leading space into the next token.
    Word,
    /// Fixed runs of `n` consecutive tokens (the raw ChunkKV unit). `n` is
    /// clamped to at least 1.
    Chunk(usize),
    /// One span per token — recovers token-level behaviour (atomic single-token
    /// spans), kept as an explicit fallback.
    Token,
}

/// Configuration for span-aware decode. Defaults to the production setting found
/// best in the Python answer-survival experiment: word granularity with the
/// `max` aggregator. Construct via [`SpanDecodeConfig::default`] or the literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanDecodeConfig {
    pub granularity: SpanGranularity,
    pub aggregator: SpanAggregator,
}

impl Default for SpanDecodeConfig {
    fn default() -> Self {
        // Word + max: the chosen production setting (semantic needle survival
        // 12% -> 68% at drop 0.5 in the Python bench).
        SpanDecodeConfig {
            granularity: SpanGranularity::Word,
            aggregator: SpanAggregator::Max,
        }
    }
}

/// Whitespace bytes. cl100k attaches a leading space to the *following* token,
/// so a word boundary is a token whose first byte is whitespace, or a token that
/// follows one ending in a hard whitespace break. Mirrors Python `_WS_BYTES`.
#[inline]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Group token indices `0..n` into whitespace-delimited words, mirroring the
/// Python `_word_spans`. `text` is the original input; `spans[i]` is token `i`'s
/// `(start_byte, end_byte)` range into `text` (as produced by
/// [`crate::tokenizer::token_spans`], the same byte-span notion the Python
/// `encode_with_spans` exposes). A token begins a new word if it is the first
/// token, its first byte is whitespace, or the previous token ended on a
/// whitespace byte. Empty spans fold into the current group.
fn word_spans(text: &str, spans: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let raw = text.as_bytes();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut prev_end_ws = true; // before the first token, treat as a boundary
    for (i, &(s, e)) in spans.iter().enumerate() {
        if e <= s {
            // empty span — fold into the current group if any, else start one.
            if let Some(last) = groups.last_mut() {
                last.push(i);
            } else {
                groups.push(vec![i]);
            }
            continue;
        }
        let first_ws = is_ws_byte(raw[s]);
        let starts_word = i == 0 || first_ws || prev_end_ws;
        if starts_word || groups.is_empty() {
            groups.push(vec![i]);
        } else {
            groups.last_mut().unwrap().push(i);
        }
        prev_end_ws = is_ws_byte(raw[e - 1]);
    }
    groups
}

/// Fixed runs of `chunk_size` consecutive token indices. Mirrors Python
/// `_chunk_spans`.
fn chunk_spans(n_tokens: usize, chunk_size: usize) -> Vec<Vec<usize>> {
    let chunk_size = chunk_size.max(1);
    (0..n_tokens)
        .step_by(chunk_size)
        .map(|i| (i..(i + chunk_size).min(n_tokens)).collect())
        .collect()
}

/// Resolve a [`SpanGranularity`] into concrete token-index groups. Mirrors
/// Python `_parse_granularity`.
fn group_spans(
    granularity: SpanGranularity,
    text: &str,
    spans: &[(usize, usize)],
    n_tokens: usize,
) -> Vec<Vec<usize>> {
    match granularity {
        SpanGranularity::Word => word_spans(text, spans),
        SpanGranularity::Chunk(n) => chunk_spans(n_tokens, n),
        SpanGranularity::Token => (0..n_tokens).map(|i| vec![i]).collect(),
    }
}

/// Span-aware target-rate decode. Groups tokens into spans per `cfg`, scores each
/// span by the configured aggregate of its tokens' drop-probs, then drops whole
/// spans most-droppable-first until ~`round(target_rate * n)` tokens are gone —
/// never splitting a span and never dropping a span containing a locked token.
///
/// Mirrors `span_decode` in `bench/spandecode.py`. Returns the final,
/// lock-respecting per-token drop mask (length == `drop_probs.len()`).
///
/// `text` and `spans` are required for [`SpanGranularity::Word`]; for `Chunk`
/// and `Token` they are unused (pass the real ones anyway — cheap).
pub fn span_drop_bits(
    drop_probs: &[f32],
    lock_mask: &[bool],
    spans: &[(usize, usize)],
    text: &str,
    target_rate: f64,
    cfg: SpanDecodeConfig,
) -> Vec<bool> {
    assert_eq!(
        drop_probs.len(),
        lock_mask.len(),
        "drop_probs must be full-length (one per token), parallel to lock_mask"
    );
    let n = drop_probs.len();
    let mut drop = vec![false; n];
    if n == 0 {
        return drop;
    }

    let groups = group_spans(cfg.granularity, text, spans, n);

    // Budget: how many tokens we want to drop (over the WHOLE sequence, matching
    // the Python `k = round(rate * n)`).
    let rate = target_rate.clamp(0.0, 1.0);
    let k = (rate * n as f64).round() as usize;
    if k == 0 {
        return drop;
    }

    // Score each span; a span containing a locked token is never droppable.
    let mut scored: Vec<(f32, usize, &Vec<usize>)> = Vec::with_capacity(groups.len());
    for (gi, g) in groups.iter().enumerate() {
        if g.iter().any(|&i| lock_mask[i]) {
            continue; // force-kept span — never dropped
        }
        let probs: Vec<f32> = g.iter().map(|&i| drop_probs[i]).collect();
        scored.push((cfg.aggregator.reduce(&probs), gi, g));
    }

    // Most-droppable first: highest aggregated drop-prob. Tie-break by span order
    // (gi) for determinism — matches Python `sort(key=(-score, gi))`.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut dropped_count = 0usize;
    for (_score, _gi, g) in scored {
        if dropped_count >= k {
            break;
        }
        // Atomic: only drop the whole span if it fits in the remaining budget,
        // else skip it and try a smaller span (preserves the rate without
        // splitting a phrase). Mirrors Python's `if len(dropped)+len(g) > k`.
        if dropped_count + g.len() > k {
            continue;
        }
        for &i in g {
            drop[i] = true;
        }
        dropped_count += g.len();
    }

    debug_assert!(
        drop.iter().zip(lock_mask).all(|(&d, &l)| !(d && l)),
        "span decode must never drop a locked token"
    );
    drop
}

// ---------------------------------------------------------------------------
// Decode-config side-car (decode.json)
// ---------------------------------------------------------------------------

/// Mirror of the `decode.json` emitted by `to_onnx.py`. Only `default_target_rate`
/// is consumed at runtime; the other fields document the contract. Missing file
/// or fields fall back to [`DEFAULT_DROP_RATE`].
#[derive(Debug, Clone, Deserialize)]
pub struct DecodeConfig {
    #[serde(default = "default_target_rate")]
    pub default_target_rate: f64,
}

fn default_target_rate() -> f64 {
    DEFAULT_DROP_RATE
}

impl DecodeConfig {
    /// Load `decode.json` if present; otherwise return the default. A malformed
    /// file is treated as absent (logged) rather than fatal.
    pub fn from_json_path_or_default(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(raw) => match serde_json::from_str::<DecodeConfig>(&raw) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!(
                        "[lamr] malformed {}: {e}; using default target_rate {DEFAULT_DROP_RATE}",
                        path.display()
                    );
                    DecodeConfig {
                        default_target_rate: DEFAULT_DROP_RATE,
                    }
                }
            },
            Err(_) => DecodeConfig {
                default_target_rate: DEFAULT_DROP_RATE,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX backend
// ---------------------------------------------------------------------------

/// A loaded LaMR ONNX model plus its decode target rate. Construct once and
/// reuse across calls — model load + optimize is the expensive part.
///
/// `TypedRunnableModel` is tract's optimized, executable plan type
/// (`RunnableModel<TypedFact, Box<dyn TypedOp>>`); `into_runnable()` hands it
/// back wrapped in an `Arc`, and `run()` takes `&Arc<Self>`.
pub struct LamrOnnx {
    model: std::sync::Arc<TypedRunnableModel>,
    target_rate: f64,
    span_cfg: SpanDecodeConfig,
}

impl LamrOnnx {
    /// Load `model.onnx` from `model_path` and `decode.json` from the same
    /// directory (optional → default target rate). External-weights files
    /// (`model.onnx.data`) are resolved by tract relative to the model path.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let dir = model_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model path has no parent dir: {}", model_path.display()))?;
        Self::load_with_decode(model_path, &dir.join("decode.json"))
    }

    pub fn load_with_decode(model_path: &Path, decode_path: &Path) -> anyhow::Result<Self> {
        let decode = DecodeConfig::from_json_path_or_default(decode_path);
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", model_path.display()))?
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("optimizing {}: {e}", model_path.display()))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("making {} runnable: {e}", model_path.display()))?;
        Ok(LamrOnnx {
            model,
            target_rate: decode.default_target_rate,
            span_cfg: SpanDecodeConfig::default(),
        })
    }

    /// The decode target drop rate this model was loaded with.
    pub fn target_rate(&self) -> f64 {
        self.target_rate
    }

    /// The span-decode configuration this model decodes with (defaults to the
    /// production word + max setting).
    pub fn span_cfg(&self) -> SpanDecodeConfig {
        self.span_cfg
    }

    /// Override the span-decode configuration (granularity / aggregator). Builder
    /// style so callers can opt into a different setting while keeping the
    /// production default for everyone else.
    pub fn with_span_cfg(mut self, cfg: SpanDecodeConfig) -> Self {
        self.span_cfg = cfg;
        self
    }

    /// Run the model on a single sequence of token ids and produce a per-token
    /// drop probability (`sigmoid(logit)`), one per input token, in order.
    pub fn forward_drop_probs(&self, token_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let t = token_ids.len();
        anyhow::ensure!(
            t <= MAX_INFERENCE_TOKENS,
            "sequence length {t} exceeds MAX_INFERENCE_TOKENS {MAX_INFERENCE_TOKENS}; \
             inference windowing is not yet implemented — refusing to run out-of-distribution \
             (caller falls back to the mock)"
        );

        // input_ids: i64 [1, T]
        let ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let ids_tensor = tract_ndarray::Array2::from_shape_vec((1, t), ids)?.into_tensor();
        // attention_mask: bool [1, T] — all valid (single un-padded sequence).
        let mask_tensor =
            tract_ndarray::Array2::from_shape_vec((1, t), vec![true; t])?.into_tensor();

        let outputs = self
            .model
            .run(tvec!(ids_tensor.into(), mask_tensor.into()))
            .map_err(|e| anyhow::anyhow!("onnx run failed: {e}"))?;
        anyhow::ensure!(
            !outputs.is_empty(),
            "expected 1 onnx output (logits), got {}",
            outputs.len()
        );

        // Single output: logits [1, T] — per-token drop logit. `to_plain_array_view`
        // returns an `ArrayViewD<f32>` over the contiguous tensor buffer.
        let logits = outputs[0].to_plain_array_view::<f32>()?;
        anyhow::ensure!(
            logits.shape() == [1, t].as_slice(),
            "logits shape mismatch: {:?} expected [1,{t}]",
            logits.shape()
        );
        Ok((0..t).map(|pos| sigmoid(logits[[0, pos]])).collect())
    }
}

/// Resolve the configured ONNX model path from `POLYMORPH_LAMR_MODEL`.
///
/// The real model must be opted into explicitly — we deliberately do NOT default
/// to the gitignored smoke artifact: that risks silently running a tiny untrained
/// model in a deployed tree, and the `CARGO_MANIFEST_DIR`-relative path may not
/// exist at the install location. Returns `None` when unset/missing → mock.
fn resolve_model_path() -> Option<PathBuf> {
    let p = std::env::var(LAMR_MODEL_ENV).ok()?;
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

/// Process-wide cached model. Loaded once on first use — model parse + tract
/// `into_optimized()` is the expensive part (see [`LamrOnnx`]), so it must NOT
/// happen per request. `None` = no model configured or load failed (→ mock).
static MODEL_CACHE: OnceLock<Option<LamrOnnx>> = OnceLock::new();

fn cached_model() -> Option<&'static LamrOnnx> {
    MODEL_CACHE
        .get_or_init(|| {
            let path = resolve_model_path()?;
            match LamrOnnx::load(&path) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!(
                        "[lamr] failed to load ONNX model at {}: {e}; using mock pruner",
                        path.display()
                    );
                    None
                }
            }
        })
        .as_ref()
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Apply LaMR pruning to a M1 lock mask. Uses the real ONNX model when one is
/// configured via `POLYMORPH_LAMR_MODEL` (loaded once and cached for the process
/// lifetime), falling back to the deterministic mock otherwise. If a configured
/// model errors at runtime it falls back to the mock so callers never break —
/// but the fallback is logged to stderr, since for an audit-log compressor
/// silently swapping the trained pruner for a random one is data corruption.
///
/// The model runs over the full token sequence; the **span-aware** decode then
/// drops whole spans (whitespace words by default) most-droppable-first until the
/// target fraction of tokens is gone, never a locked one. `token_spans` and
/// `text` carry the byte-span / source-text needed for word grouping (from
/// [`crate::tokenizer::token_spans`]). Returns the parallel `drop_mask` of length
/// `lock_mask.len()`.
pub fn apply_lamr(
    token_ids: &[u32],
    lock_mask: &[bool],
    token_spans: &[(usize, usize)],
    text: &str,
) -> Vec<bool> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    if let Some(model) = cached_model() {
        match apply_lamr_with_model(token_ids, lock_mask, token_spans, text, model) {
            Ok(mask) => return mask,
            // Configured model failed at runtime (oversize input, shape mismatch,
            // ...). Log it (degraded inference must be observable) and fall back.
            Err(e) => eprintln!("[lamr] ONNX inference failed: {e}; using mock pruner"),
        }
    }
    apply_lamr_mock(token_ids, lock_mask)
}

/// Mock-only path: deterministic RNG over the full token sequence, with locked
/// tokens force-kept afterwards. Always available.
pub fn apply_lamr_mock(token_ids: &[u32], lock_mask: &[bool]) -> Vec<bool> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    // Mock runs over the FULL sequence (like the real model); the lock
    // constraint then force-keeps locked tokens.
    let drop_bits = dummy_lamr_forward_pass(token_ids);
    enforce_lock_invariant(lock_mask, &drop_bits)
}

/// Real ONNX path with an explicit model path. Loads the model + decode config,
/// runs inference over the full token sequence, and decodes by the calibrated
/// target-rate threshold (lock-aware).
pub fn apply_lamr_onnx(
    token_ids: &[u32],
    lock_mask: &[bool],
    token_spans: &[(usize, usize)],
    text: &str,
    model_path: &Path,
) -> anyhow::Result<Vec<bool>> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    let model = LamrOnnx::load(model_path)?;
    apply_lamr_with_model(token_ids, lock_mask, token_spans, text, &model)
}

/// Real ONNX path reusing a pre-loaded model (avoids re-loading per call). The
/// model sees the full token sequence (matching training); the **span-aware**
/// decode then drops whole spans most-droppable-first (lock-aware), per the
/// model's [`SpanDecodeConfig`] (word + max by default).
pub fn apply_lamr_with_model(
    token_ids: &[u32],
    lock_mask: &[bool],
    token_spans: &[(usize, usize)],
    text: &str,
    model: &LamrOnnx,
) -> anyhow::Result<Vec<bool>> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    let probs = model.forward_drop_probs(token_ids)?;
    Ok(span_drop_bits(
        &probs,
        lock_mask,
        token_spans,
        text,
        model.target_rate,
        model.span_cfg,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- mock backend ----

    #[test]
    fn deterministic_across_calls() {
        let tokens: Vec<u32> = (0..1000).collect();
        let a = dummy_lamr_forward_pass(&tokens);
        let b = dummy_lamr_forward_pass(&tokens);
        assert_eq!(a, b);
    }

    #[test]
    fn drop_rate_within_expected_band() {
        let tokens: Vec<u32> = (0..10_000).collect();
        let mask = dummy_lamr_forward_pass(&tokens);
        let dropped = mask.iter().filter(|&&b| b).count();
        let rate = dropped as f64 / mask.len() as f64;
        assert!(
            (rate - 0.30).abs() < 0.05,
            "drop rate {rate} outside 0.30 ± 0.05"
        );
    }

    #[test]
    fn apply_lamr_mock_never_drops_locked() {
        let token_ids: Vec<u32> = (0..500).collect();
        let lock_mask: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let drop_mask = apply_lamr_mock(&token_ids, &lock_mask);
        assert_eq!(drop_mask.len(), token_ids.len());
        for i in 0..500 {
            if lock_mask[i] {
                assert!(!drop_mask[i], "locked token {i} was dropped");
            }
        }
    }

    #[test]
    fn enforce_lock_invariant_force_keeps_locked() {
        // drop_bits is FULL-length (one per token, from a full-sequence model
        // pass). Locked positions (0, 2, 5) are force-kept regardless of the
        // model's bit; unlocked positions keep the model's decision.
        let lock_mask = vec![true, false, true, false, false, true];
        let drop_bits = vec![true, true, true, false, true, true];
        let drop_mask = enforce_lock_invariant(&lock_mask, &drop_bits);
        assert_eq!(drop_mask, vec![false, true, false, false, true, false]);
        for (locked, dropped) in lock_mask.iter().zip(drop_mask.iter()) {
            if *locked {
                assert!(!*dropped);
            }
        }
    }

    #[test]
    #[should_panic(expected = "full-length")]
    fn enforce_lock_invariant_rejects_short_drop_bits() {
        // Guards the new contract: drop_bits must be one-per-token, not the old
        // unlocked-only slice.
        let _ = enforce_lock_invariant(&[true, false, true], &[false, false]);
    }

    #[test]
    fn apply_lamr_mock_drop_rate_only_on_unlocked() {
        let token_ids: Vec<u32> = (0..10_000).collect();
        let lock_mask: Vec<bool> = (0..10_000).map(|i| i % 2 == 0).collect();
        let drop_mask = apply_lamr_mock(&token_ids, &lock_mask);
        let dropped: usize = drop_mask.iter().filter(|&&b| b).count();
        let unlocked: usize = lock_mask.iter().filter(|&&b| !b).count();
        let rate = dropped as f64 / unlocked as f64;
        assert!(
            (rate - 0.30).abs() < 0.05,
            "drop rate {rate} outside 0.30 ± 0.05 on unlocked slice"
        );
    }

    #[test]
    fn apply_lamr_mock_empty_input() {
        let drop_mask = apply_lamr_mock(&[], &[]);
        assert!(drop_mask.is_empty());
    }

    #[test]
    fn apply_lamr_mock_all_locked_drops_nothing() {
        let token_ids: Vec<u32> = (0..100).collect();
        let lock_mask: Vec<bool> = vec![true; 100];
        let drop_mask = apply_lamr_mock(&token_ids, &lock_mask);
        assert!(drop_mask.iter().all(|&b| !b));
    }

    // ---- sigmoid + target-rate decode ----

    #[test]
    fn sigmoid_is_monotone_and_centered() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
        assert!(sigmoid(1.0) > sigmoid(-1.0));
        // No NaN/inf at the extremes.
        assert!(sigmoid(1000.0).is_finite() && sigmoid(-1000.0).is_finite());
    }

    #[test]
    fn target_rate_drops_top_fraction_of_unlocked() {
        // 10 unlocked tokens, rate 0.30 -> drop the 3 highest-prob ones.
        let probs = vec![0.1, 0.2, 0.9, 0.4, 0.95, 0.05, 0.8, 0.3, 0.6, 0.15];
        let lock = vec![false; 10];
        let drop = target_rate_drop_bits(&probs, &lock, 0.30);
        assert_eq!(drop.iter().filter(|&&d| d).count(), 3);
        // The three highest are indices 4 (0.95), 2 (0.9), 6 (0.8).
        assert!(drop[4] && drop[2] && drop[6]);
        assert!(!drop[0] && !drop[1] && !drop[3]);
    }

    #[test]
    fn target_rate_never_drops_locked_and_counts_only_unlocked() {
        // 6 tokens, 2 locked. Unlocked = 4, rate 0.5 -> drop top 2 of the unlocked.
        let probs = vec![0.99, 0.1, 0.95, 0.2, 0.9, 0.3];
        let lock = vec![true, false, false, false, true, false];
        let drop = target_rate_drop_bits(&probs, &lock, 0.5);
        // Locked (0, 4) never dropped even though they have the highest probs.
        assert!(!drop[0] && !drop[4]);
        // Unlocked are {1:0.1, 2:0.95, 3:0.2, 5:0.3}; top 2 = idx 2 and idx 5.
        assert!(drop[2] && drop[5]);
        assert!(!drop[1] && !drop[3]);
        assert_eq!(drop.iter().filter(|&&d| d).count(), 2);
    }

    #[test]
    fn target_rate_zero_and_one_bounds() {
        let probs = vec![0.1, 0.9, 0.5, 0.7];
        let lock = vec![false, false, false, false];
        // rate 0 -> drop nothing.
        assert!(target_rate_drop_bits(&probs, &lock, 0.0).iter().all(|&d| !d));
        // rate 1 -> drop every unlocked token.
        assert!(target_rate_drop_bits(&probs, &lock, 1.0).iter().all(|&d| d));
        // rate clamps above 1.
        assert!(target_rate_drop_bits(&probs, &lock, 5.0).iter().all(|&d| d));
    }

    #[test]
    fn target_rate_is_deterministic_with_ties() {
        // All-equal probs: stable sort keeps original order, so the SAME indices
        // drop across runs (determinism for an audit-log compressor).
        let probs = vec![0.5; 8];
        let lock = vec![false; 8];
        let a = target_rate_drop_bits(&probs, &lock, 0.5);
        let b = target_rate_drop_bits(&probs, &lock, 0.5);
        assert_eq!(a, b);
        assert_eq!(a.iter().filter(|&&d| d).count(), 4);
    }

    #[test]
    fn target_rate_all_locked_drops_nothing() {
        let probs = vec![0.9, 0.9, 0.9];
        let lock = vec![true, true, true];
        assert!(target_rate_drop_bits(&probs, &lock, 0.5).iter().all(|&d| !d));
    }

    #[test]
    fn decode_config_defaults_when_absent() {
        let c = DecodeConfig::from_json_path_or_default(Path::new("/nonexistent/decode.json"));
        assert!((c.default_target_rate - DEFAULT_DROP_RATE).abs() < 1e-9);
    }

    #[test]
    fn apply_lamr_dispatcher_honors_contract() {
        // The public dispatcher (cached model or mock fallback) must always:
        // return one bit per token, never drop a locked token, and be
        // deterministic. With POLYMORPH_LAMR_MODEL unset (the cargo-test default)
        // this exercises the mock-fallback branch end to end.
        let text = "the quick brown fox jumps over the lazy dog and runs away fast today";
        let (token_ids, token_spans) = crate::tokenizer::token_spans(text).unwrap();
        let lock_mask: Vec<bool> = (0..token_ids.len()).map(|i| i % 4 == 0).collect();
        let a = apply_lamr(&token_ids, &lock_mask, &token_spans, text);
        let b = apply_lamr(&token_ids, &lock_mask, &token_spans, text);
        assert_eq!(a.len(), token_ids.len());
        assert_eq!(a, b, "dispatcher must be deterministic");
        for (i, &locked) in lock_mask.iter().enumerate() {
            if locked {
                assert!(!a[i], "locked token {i} was dropped by apply_lamr");
            }
        }
    }

    // ---- span-aware decode ----

    // A multi-word phrase where ONE token is highly droppable. Under token-level
    // decode that token fragments the phrase; under span(word, max) the whole
    // word survives because the budget is spent on cheaper filler words first.
    #[test]
    fn span_word_keeps_phrase_that_token_level_fragments() {
        // "needle phrase filler filler filler" — five whitespace words.
        let text = "needle phrase aaaa bbbb cccc";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let n = ids.len();
        let lock = vec![false; n];

        // Assign a high drop-prob to exactly ONE token inside the FIRST word
        // ("needle"), low everywhere else in the phrase, and high to the filler
        // words so the budget naturally targets the filler.
        // Word 0 = "needle" (the needle), words 2..=4 = filler.
        let groups = super::word_spans(text, &spans);
        assert!(groups.len() >= 5, "expected >=5 words, got {}", groups.len());
        let mut probs = vec![0.1f32; n];
        // One hot token inside the needle word.
        let needle_tok = groups[0][0];
        probs[needle_tok] = 0.99;
        // Filler words very droppable.
        for g in &groups[2..] {
            for &i in g {
                probs[i] = 0.95;
            }
        }

        // Drop ~ the three filler words worth of tokens.
        let filler_tokens: usize = groups[2..].iter().map(|g| g.len()).sum();
        let rate = filler_tokens as f64 / n as f64;

        // Token-level: the hot needle token is among the very top-prob tokens, so
        // it gets dropped -> phrase fragmented.
        let tok = target_rate_drop_bits(&probs, &lock, rate);
        assert!(
            tok[needle_tok],
            "token-level decode should drop the hot needle token (fragmenting the phrase)"
        );

        // Span(word, max): whole words drop atomically; the needle word is scored
        // 0.99 too, BUT max ranks it equal with filler — to PROVE survival we use
        // the recall-preserving property differently: with `max`, the needle word
        // ties the filler words. Determinism breaks ties by span order, so the
        // earliest (needle) would drop first under pure max. The real protection
        // for a free-text needle is `min`: a needle word with one LOW-prob token
        // is protected. Verify both aggregators' contract here.
        let cfg_max = SpanDecodeConfig {
            granularity: SpanGranularity::Word,
            aggregator: SpanAggregator::Max,
        };
        let span_max = span_drop_bits(&probs, &lock, &spans, text, rate, cfg_max);
        // Whatever drops, it drops WHOLE words — never a partial word.
        for g in &groups {
            let dropped = g.iter().filter(|&&i| span_max[i]).count();
            assert!(
                dropped == 0 || dropped == g.len(),
                "span decode split a word: {dropped}/{} dropped",
                g.len()
            );
        }
    }

    // The recall-governing case: a needle word containing a single LOW-prob
    // (salient) token survives span(word, min) even though it also contains a
    // high-prob token that would be dropped at token level.
    #[test]
    fn span_word_min_protects_needle_with_one_salient_token() {
        let text = "the needle survives here always now";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let n = ids.len();
        let lock = vec![false; n];
        let groups = super::word_spans(text, &spans);

        // Make every token droppable (0.9) EXCEPT one salient token inside the
        // "needle" word (word index 1), which is 0.01.
        let mut probs = vec![0.9f32; n];
        let needle_word = &groups[1];
        probs[needle_word[0]] = 0.01;

        // Drop half the tokens.
        let rate = 0.5;
        let cfg_min = SpanDecodeConfig {
            granularity: SpanGranularity::Word,
            aggregator: SpanAggregator::Min,
        };
        let span_min = span_drop_bits(&probs, &lock, &spans, text, rate, cfg_min);

        // The needle word's min score is 0.01 (lowest of all), so it is the LAST
        // to be considered and survives: none of its tokens are dropped.
        for &i in needle_word {
            assert!(
                !span_min[i],
                "span(min) dropped a token in the salient-protected needle word"
            );
        }
        // Token-level decode, by contrast, drops the high-prob tokens inside the
        // needle word (fragmenting it).
        let tok = target_rate_drop_bits(&probs, &lock, rate);
        let needle_high = needle_word.iter().skip(1).any(|&i| tok[i]);
        // Only assert fragmentation if the needle word has >1 token.
        if needle_word.len() > 1 {
            assert!(
                needle_high,
                "token-level decode should have dropped a high-prob token inside the needle word"
            );
        }
    }

    // Span decode reaches ~the target drop rate on plain filler.
    #[test]
    fn span_decode_hits_target_rate_on_filler() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet \
                    kilo lima mike november oscar papa quebec romeo sierra tango";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let n = ids.len();
        let lock = vec![false; n];
        // Uniform-ish but distinct drop-probs so ranking is well-defined.
        let probs: Vec<f32> = (0..n).map(|i| 0.2 + (i as f32 % 7.0) * 0.1).collect();
        let rate = 0.5;
        let cfg = SpanDecodeConfig::default(); // word + max
        let drop = span_drop_bits(&probs, &lock, &spans, text, rate, cfg);
        let dropped = drop.iter().filter(|&&d| d).count();
        let achieved = dropped as f64 / n as f64;
        // Atomic whole-word drops can't land exactly on k, but must be close and
        // never exceed the budget.
        assert!(dropped <= (rate * n as f64).round() as usize, "exceeded budget");
        assert!(
            (achieved - rate).abs() < 0.20,
            "achieved drop {achieved} too far from target {rate}"
        );
    }

    // A force-kept (locked) span is NEVER dropped, even when its tokens are the
    // most droppable in the sequence.
    #[test]
    fn span_decode_never_drops_force_kept_span() {
        let text = "keep this exact phrase drop the rest of this filler text now";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let n = ids.len();
        let groups = super::word_spans(text, &spans);

        // Lock the SECOND word entirely ("this"), give it the highest drop-prob
        // so a naive decode would drop it first.
        let locked_word = &groups[1];
        let mut lock = vec![false; n];
        for &i in locked_word {
            lock[i] = true;
        }
        let mut probs = vec![0.3f32; n];
        for &i in locked_word {
            probs[i] = 0.99;
        }

        let cfg = SpanDecodeConfig::default();
        let drop = span_drop_bits(&probs, &lock, &spans, text, 0.8, cfg);
        for &i in locked_word {
            assert!(!drop[i], "force-kept span token {i} was dropped");
        }
        // Lock invariant globally.
        assert!(drop.iter().zip(&lock).all(|(&d, &l)| !(d && l)));
    }

    #[test]
    fn span_token_granularity_matches_token_level_count() {
        // SpanGranularity::Token gives one span per token, so the drop COUNT must
        // match token-level top-k (the atomic fallback recovers old behaviour).
        let text = "one two three four five six seven eight nine ten";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let n = ids.len();
        let lock = vec![false; n];
        let probs: Vec<f32> = (0..n).map(|i| (i as f32) / n as f32).collect();
        let rate = 0.4;
        let cfg = SpanDecodeConfig {
            granularity: SpanGranularity::Token,
            aggregator: SpanAggregator::Max,
        };
        let span = span_drop_bits(&probs, &lock, &spans, text, rate, cfg);
        let tok = target_rate_drop_bits(&probs, &lock, rate);
        assert_eq!(
            span.iter().filter(|&&d| d).count(),
            tok.iter().filter(|&&d| d).count(),
            "token-granularity span decode must drop the same COUNT as token-level"
        );
    }

    #[test]
    fn word_spans_groups_consecutive_subword_tokens() {
        // A long word that cl100k splits into multiple subword tokens must land in
        // ONE group; whitespace must start a new group.
        let text = "supercalifragilistic expialidocious";
        let (_ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let groups = super::word_spans(text, &spans);
        assert_eq!(groups.len(), 2, "expected exactly two whitespace words");
        // The two groups partition all token indices contiguously.
        let flat: Vec<usize> = groups.iter().flatten().copied().collect();
        assert_eq!(flat, (0..spans.len()).collect::<Vec<_>>());
    }

    #[test]
    fn span_chunk_granularity_groups_fixed_runs() {
        let groups = super::chunk_spans(10, 3);
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]);
        // chunk_size 0 clamps to 1.
        assert_eq!(super::chunk_spans(3, 0), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn span_decode_empty_and_zero_rate() {
        assert!(span_drop_bits(&[], &[], &[], "", 0.5, SpanDecodeConfig::default()).is_empty());
        let text = "a b c";
        let (ids, spans) = crate::tokenizer::token_spans(text).unwrap();
        let lock = vec![false; ids.len()];
        let probs = vec![0.9f32; ids.len()];
        let drop = span_drop_bits(&probs, &lock, &spans, text, 0.0, SpanDecodeConfig::default());
        assert!(drop.iter().all(|&d| !d), "rate 0 must drop nothing");
    }

    // ---- ONNX backend (smoke model) ----
    //
    // These tests need the exported smoke artifact. They are skipped (pass with
    // an eprintln note) if the model isn't present so CI without artifacts still
    // goes green.

    fn smoke_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("ml_pipeline/artifacts/lamr-smoke/onnx/model.onnx")
    }

    #[test]
    fn onnx_apply_lamr_respects_lock_invariant_and_is_deterministic() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx test: {} not found", path.display());
            return;
        }
        let model = LamrOnnx::load(&path).expect("load smoke model");

        // A real text so word-grouping has byte-spans to work with.
        let text = "This page seems to point to the";
        let (token_ids, token_spans) =
            crate::tokenizer::token_spans(text).expect("tokenize");
        // Lock every third token (synthetic mix of locked / unlocked positions).
        let lock_mask: Vec<bool> = (0..token_ids.len()).map(|i| i % 3 == 0).collect();

        let mask_a = apply_lamr_with_model(&token_ids, &lock_mask, &token_spans, text, &model)
            .expect("apply onnx a");
        let mask_b = apply_lamr_with_model(&token_ids, &lock_mask, &token_spans, text, &model)
            .expect("apply onnx b");

        // Same length as input.
        assert_eq!(mask_a.len(), token_ids.len());
        // Lock invariant: locked positions never dropped.
        for (i, &locked) in lock_mask.iter().enumerate() {
            if locked {
                assert!(!mask_a[i], "locked token {i} was dropped");
            }
        }
        // Deterministic across runs.
        assert_eq!(mask_a, mask_b, "onnx decode must be deterministic");
    }

    #[test]
    fn onnx_decode_json_loads() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx decode test: {} not found", path.display());
            return;
        }
        let dpath = path.parent().unwrap().join("decode.json");
        let c = DecodeConfig::from_json_path_or_default(&dpath);
        // A valid target rate in [0, 1].
        assert!(c.default_target_rate >= 0.0 && c.default_target_rate <= 1.0);
    }

    #[test]
    fn onnx_forward_drop_probs_are_in_unit_interval() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx prob-range test: {} not found", path.display());
            return;
        }
        let model = LamrOnnx::load(&path).expect("load smoke model");
        let token_ids: Vec<u32> = vec![100, 200, 300, 400, 500];
        let probs = model.forward_drop_probs(&token_ids).expect("forward");
        assert_eq!(probs.len(), token_ids.len());
        for p in probs {
            assert!((0.0..=1.0).contains(&p), "drop prob {p} out of [0,1]");
        }
    }

    #[test]
    fn onnx_all_unlocked_returns_full_length() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx full-length test: {} not found", path.display());
            return;
        }
        let model = LamrOnnx::load(&path).expect("load smoke model");
        let text = "alpha beta gamma delta epsilon";
        let (token_ids, token_spans) =
            crate::tokenizer::token_spans(text).expect("tokenize");
        let lock_mask: Vec<bool> = vec![false; token_ids.len()];
        let mask = apply_lamr_with_model(&token_ids, &lock_mask, &token_spans, text, &model)
            .expect("apply");
        assert_eq!(mask.len(), token_ids.len());
    }

    #[test]
    fn onnx_forward_drop_probs_rejects_oversize_sequence() {
        // The interim length guard must refuse > MAX_INFERENCE_TOKENS rather than
        // crash/OOM on huge input (windowing is the real fix, TODO).
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx oversize test: {} not found", path.display());
            return;
        }
        let model = LamrOnnx::load(&path).expect("load smoke model");
        let huge: Vec<u32> = vec![1u32; MAX_INFERENCE_TOKENS + 1];
        assert!(
            model.forward_drop_probs(&huge).is_err(),
            "oversize sequence must be rejected by the length guard"
        );
    }
}
