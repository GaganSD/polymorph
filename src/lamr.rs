//! LaMR pruner.
//!
//! Two backends live here:
//!   * a deterministic RNG **mock** (`dummy_lamr_forward_pass`), kept as the
//!     fallback so callers/tests work without an exported model, and
//!   * the **real** ONNX-backed pruner (`apply_lamr_onnx`), which loads the
//!     LaMR model exported by `polymorph_lamr/export/to_onnx.py` (backbone + one
//!     emission head → an `emissions` tensor), pairs it with the single
//!     linear-chain CRF transition set from `transitions.json`, and runs a
//!     Viterbi decode in Rust to produce the per-token drop bits.
//!
//! Both backends run the model over the **full** token sequence (matching how
//! the labeler tags every token at training time), then funnel through
//! [`enforce_lock_invariant`], which projects the deterministic lock constraint:
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

/// Target drop rate for the mock. The mock returns true (drop) with this
/// probability for each unlocked token.
pub const DEFAULT_DROP_RATE: f64 = 0.30;

/// Number of CRF tags. Tag 0 = keep, tag 1 = drop. Fixed by the exported model
/// (see `polymorph_lamr/model/crf.py`, `NUM_TAGS = 2`).
const NUM_TAGS: usize = 2;

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

/// Project the lock constraint onto the model's per-token drop decisions.
///
/// The neural model runs over the **full** token sequence — exactly as it was
/// trained (the labeler tags every token of the chunk), so the encoder sees the
/// same context at train and inference time. `drop_bits[i]` is the model's
/// decision for token `i` (`true` = drop). A locked token is then force-kept
/// here: the deterministic lock is a hard must-keep constraint applied *after*
/// the model and never shown to it.
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
// CRF transition params (side-car JSON)
// ---------------------------------------------------------------------------

/// Mirror of `transitions.json` emitted by `to_onnx.py` for the single CRF.
/// `trans` is a 2x2 row-major matrix where `trans[i][j]` is the score of moving
/// from tag `i` to tag `j`; `start` / `end` are length-2 vectors.
#[derive(Debug, Clone, Deserialize)]
pub struct Transitions {
    pub trans: Vec<Vec<f32>>,
    pub start: Vec<f32>,
    pub end: Vec<f32>,
}

impl Transitions {
    pub fn from_json_path(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let t: Transitions = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
        t.validate()?;
        Ok(t)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.trans.len() == NUM_TAGS && self.trans.iter().all(|r| r.len() == NUM_TAGS),
            "trans must be {NUM_TAGS}x{NUM_TAGS}"
        );
        anyhow::ensure!(self.start.len() == NUM_TAGS, "start must have len {NUM_TAGS}");
        anyhow::ensure!(self.end.len() == NUM_TAGS, "end must have len {NUM_TAGS}");
        Ok(())
    }
}

/// One linear-chain CRF parameter set, ready for Viterbi.
#[derive(Debug, Clone)]
pub struct Crf {
    /// `transitions[i][j]` = score(tag i -> tag j).
    pub transitions: [[f32; NUM_TAGS]; NUM_TAGS],
    pub start: [f32; NUM_TAGS],
    pub end: [f32; NUM_TAGS],
}

impl Crf {
    /// Pack a validated [`Transitions`] (variable-length Vecs) into fixed-size
    /// arrays for the decoder. Callers must pass a `Transitions` that has been
    /// shape-checked (`from_json_path` guarantees this); indexing assumes the
    /// `NUM_TAGS`x`NUM_TAGS` shape.
    pub fn from_transitions(t: &Transitions) -> Self {
        let mut transitions = [[0.0f32; NUM_TAGS]; NUM_TAGS];
        let mut start = [0.0f32; NUM_TAGS];
        let mut end = [0.0f32; NUM_TAGS];
        for i in 0..NUM_TAGS {
            for j in 0..NUM_TAGS {
                transitions[i][j] = t.trans[i][j];
            }
            start[i] = t.start[i];
            end[i] = t.end[i];
        }
        Crf {
            transitions,
            start,
            end,
        }
    }
}

// ---------------------------------------------------------------------------
// Linear-chain Viterbi decode
// ---------------------------------------------------------------------------

/// Linear-chain Viterbi MAP decode over `NUM_TAGS` tags.
///
/// `emissions[t][k]` is the log-emission for tag `k` at position `t`.
/// Returns the best tag sequence of length `emissions.len()`. Tag 1 = drop.
///
/// Matches the reference decode in `polymorph_lamr/model/crf.py::_viterbi`:
///   * `score[k] = start[k] + emissions[0][k]`
///   * for each subsequent position, `best_prev[k] = argmax_p (score[p] +
///     transitions[p][k])`, then `score[k] = best_score[k] + emissions[t][k]`
///   * add `end[k]`, take the argmax as the last tag, backtrack.
pub fn viterbi_decode(emissions: &[[f32; NUM_TAGS]], crf: &Crf) -> Vec<u8> {
    let t = emissions.len();
    if t == 0 {
        return Vec::new();
    }

    // score[k] = best path score ending in tag k at the current position.
    let mut score = [0.0f32; NUM_TAGS];
    for k in 0..NUM_TAGS {
        score[k] = crf.start[k] + emissions[0][k];
    }

    // backpointers[i][k] = the tag at position i that leads into tag k at i+1.
    // history has t-1 entries (one per transition step), matching the Python.
    let mut history: Vec<[usize; NUM_TAGS]> = Vec::with_capacity(t.saturating_sub(1));

    for pos in 1..t {
        let mut next_score = [f32::NEG_INFINITY; NUM_TAGS];
        let mut best_prev = [0usize; NUM_TAGS];
        for cur in 0..NUM_TAGS {
            // Pick the previous tag that maximises score[prev] + trans[prev][cur].
            let mut best_p = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for prev in 0..NUM_TAGS {
                let v = score[prev] + crf.transitions[prev][cur];
                if v > best_v {
                    best_v = v;
                    best_p = prev;
                }
            }
            best_prev[cur] = best_p;
            next_score[cur] = best_v + emissions[pos][cur];
        }
        history.push(best_prev);
        score = next_score;
    }

    // Add end transitions and pick the final tag.
    let mut best_last = 0usize;
    let mut best_last_v = f32::NEG_INFINITY;
    for k in 0..NUM_TAGS {
        let v = score[k] + crf.end[k];
        if v > best_last_v {
            best_last_v = v;
            best_last = k;
        }
    }

    // Backtrack.
    let mut tags_rev: Vec<u8> = Vec::with_capacity(t);
    let mut tag = best_last;
    tags_rev.push(tag as u8);
    for pos in (0..t - 1).rev() {
        tag = history[pos][tag];
        tags_rev.push(tag as u8);
    }
    tags_rev.reverse();
    tags_rev
}

// ---------------------------------------------------------------------------
// ONNX backend
// ---------------------------------------------------------------------------

/// A loaded LaMR ONNX model plus its CRF transition side-car. Construct once and
/// reuse across calls — model load + optimize is the expensive part.
///
/// `TypedRunnableModel` is tract's optimized, executable plan type
/// (`RunnableModel<TypedFact, Box<dyn TypedOp>>`); `into_runnable()` hands it
/// back wrapped in an `Arc`, and `run()` takes `&Arc<Self>`.
pub struct LamrOnnx {
    model: std::sync::Arc<TypedRunnableModel>,
    crf: Crf,
}

impl LamrOnnx {
    /// Load `model.onnx` from `model_path` and `transitions.json` from the same
    /// directory. External-weights files (`model.onnx.data`) are resolved by
    /// tract relative to the model path.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let dir = model_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model path has no parent dir: {}", model_path.display()))?;
        Self::load_with_transitions(model_path, &dir.join("transitions.json"))
    }

    pub fn load_with_transitions(
        model_path: &Path,
        transitions_path: &Path,
    ) -> anyhow::Result<Self> {
        let transitions = Transitions::from_json_path(transitions_path)?;
        let crf = Crf::from_transitions(&transitions);
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", model_path.display()))?
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("optimizing {}: {e}", model_path.display()))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("making {} runnable: {e}", model_path.display()))?;
        Ok(LamrOnnx { model, crf })
    }

    /// Run the model on a single sequence of token ids and produce the per-token
    /// drop bit (true = drop) via CRF Viterbi. Returns one bit per input token,
    /// in order.
    pub fn forward_drop_bits(&self, token_ids: &[u32]) -> anyhow::Result<Vec<bool>> {
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
            "expected 1 onnx output (emissions), got {}",
            outputs.len()
        );

        // Single output: emissions [1, T, 2]. `to_plain_array_view` returns an
        // `ArrayViewD<f32>` over the contiguous tensor buffer.
        let emi = outputs[0].to_plain_array_view::<f32>()?;
        anyhow::ensure!(
            emi.shape() == [1, t, NUM_TAGS].as_slice(),
            "emission shape mismatch: {:?} expected [1,{t},{NUM_TAGS}]",
            emi.shape()
        );

        // Pack the per-token emissions into the Viterbi input layout.
        let mut rows: Vec<[f32; NUM_TAGS]> = Vec::with_capacity(t);
        for pos in 0..t {
            let mut row = [0.0f32; NUM_TAGS];
            for k in 0..NUM_TAGS {
                row[k] = emi[[0, pos, k]];
            }
            rows.push(row);
        }

        let tags = viterbi_decode(&rows, &self.crf);
        Ok(tags.into_iter().map(|tag| tag == 1).collect())
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
/// The model runs over the full token sequence; the model's per-token drop
/// decision sets `drop_mask[i]`, except locked tokens (`lock_mask[i] == true`),
/// which are always kept. Returns the parallel `drop_mask` of length
/// `lock_mask.len()`.
pub fn apply_lamr(token_ids: &[u32], lock_mask: &[bool]) -> Vec<bool> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    if let Some(model) = cached_model() {
        match apply_lamr_with_model(token_ids, lock_mask, model) {
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

/// Real ONNX path with an explicit model path. Loads the model + transitions,
/// runs inference over the full token sequence, decodes the CRF with Viterbi,
/// and projects the lock constraint via [`enforce_lock_invariant`].
pub fn apply_lamr_onnx(
    token_ids: &[u32],
    lock_mask: &[bool],
    model_path: &Path,
) -> anyhow::Result<Vec<bool>> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    let model = LamrOnnx::load(model_path)?;
    apply_lamr_with_model(token_ids, lock_mask, &model)
}

/// Real ONNX path reusing a pre-loaded model (avoids re-loading per call).
pub fn apply_lamr_with_model(
    token_ids: &[u32],
    lock_mask: &[bool],
    model: &LamrOnnx,
) -> anyhow::Result<Vec<bool>> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    // The model sees the full token sequence (matching training); locked tokens
    // are force-kept afterwards by enforce_lock_invariant.
    let drop_bits = model.forward_drop_bits(token_ids)?;
    Ok(enforce_lock_invariant(lock_mask, &drop_bits))
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

    // ---- Viterbi ----

    fn flat_crf() -> Crf {
        Crf {
            transitions: [[0.0, 0.0], [0.0, 0.0]],
            start: [0.0, 0.0],
            end: [0.0, 0.0],
        }
    }

    #[test]
    fn viterbi_empty_is_empty() {
        assert!(viterbi_decode(&[], &flat_crf()).is_empty());
    }

    #[test]
    fn viterbi_picks_per_token_argmax_when_transitions_flat() {
        // With all transitions/start/end = 0, the best path is just per-token
        // argmax of the emissions.
        let crf = flat_crf();
        let emissions = vec![
            [0.1, 0.9], // -> 1 (drop)
            [0.8, 0.2], // -> 0 (keep)
            [0.3, 0.7], // -> 1 (drop)
        ];
        assert_eq!(viterbi_decode(&emissions, &crf), vec![1, 0, 1]);
    }

    #[test]
    fn viterbi_transition_overrides_weak_emission() {
        // Hand-computed 2-state example.
        //
        // emissions:  pos0 = [0, 0],  pos1 = [0, 1]   (tag1 slightly favoured at pos1)
        // start = [0, 0], end = [0, 0]
        // transitions: strongly discourage 0->1 (-10), everything else 0.
        //   trans[0][1] = -10, trans[0][0] = 0, trans[1][0] = 0, trans[1][1] = 0
        //
        // Candidate full-path scores (start+emit+trans+emit+end):
        //   [0,0]: 0 + 0   + t00(0)  + e1_0(0)  + 0 = 0
        //   [0,1]: 0 + 0   + t01(-10)+ e1_1(1)  + 0 = -9
        //   [1,0]: 0 + 0   + t10(0)  + e1_0(0)  + 0 = 0
        //   [1,1]: 0 + 0   + t11(0)  + e1_1(1)  + 0 = 1   <-- max
        // Best path = [1, 1].
        let crf = Crf {
            transitions: [[0.0, -10.0], [0.0, 0.0]],
            start: [0.0, 0.0],
            end: [0.0, 0.0],
        };
        let emissions = vec![[0.0, 0.0], [0.0, 1.0]];
        assert_eq!(viterbi_decode(&emissions, &crf), vec![1, 1]);
    }

    #[test]
    fn viterbi_start_and_end_bias() {
        // Verify start/end transitions enter the score.
        // start = [5, 0], end = [0, 5]; emissions all zero; transitions zero.
        //   [0,0]: 5 + 0 + 0 = 5
        //   [0,1]: 5 + 0 + 5 = 10  <-- max
        //   [1,0]: 0 + 0 + 0 = 0
        //   [1,1]: 0 + 0 + 5 = 5
        // Best path = [0, 1].
        let crf = Crf {
            transitions: [[0.0, 0.0], [0.0, 0.0]],
            start: [5.0, 0.0],
            end: [0.0, 5.0],
        };
        let emissions = vec![[0.0, 0.0], [0.0, 0.0]];
        assert_eq!(viterbi_decode(&emissions, &crf), vec![0, 1]);
    }

    #[test]
    fn viterbi_single_token() {
        // T=1: just start + emission + end.
        let crf = Crf {
            transitions: [[0.0, 0.0], [0.0, 0.0]],
            start: [0.0, 1.0],
            end: [0.0, 0.0],
        };
        assert_eq!(viterbi_decode(&[[0.0, 0.0]], &crf), vec![1]);
    }

    #[test]
    fn from_transitions_packs_arrays() {
        // The side-car's variable-length Vecs are packed verbatim into the
        // fixed-size decode arrays (no blending — single CRF).
        let t = Transitions {
            trans: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            start: vec![1.0, 0.0],
            end: vec![0.0, 1.0],
        };
        let c = Crf::from_transitions(&t);
        assert_eq!(c.transitions, [[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(c.start, [1.0, 0.0]);
        assert_eq!(c.end, [0.0, 1.0]);
    }

    #[test]
    fn transitions_validate_rejects_wrong_shape() {
        // A 1x2 trans matrix must be rejected by validate().
        let bad = Transitions {
            trans: vec![vec![0.0, 0.0]],
            start: vec![0.0, 0.0],
            end: vec![0.0, 0.0],
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn viterbi_golden_mixed_path() {
        // Cross-language golden: the SAME params + expected output are asserted
        // in the Python reference test (ml_pipeline tests/test_crf.py
        // test_viterbi_golden_matches_rust_decode), pinning Rust<->Python decode
        // agreement. start favors tag0, transitions favor switching, emissions
        // flat -> best path alternates: [0, 1, 0].
        let crf = Crf {
            transitions: [[-0.5, 1.0], [1.0, -0.5]],
            start: [1.0, 0.0],
            end: [0.0, 0.0],
        };
        let emissions = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert_eq!(viterbi_decode(&emissions, &crf), vec![0, 1, 0]);
    }

    #[test]
    fn apply_lamr_dispatcher_honors_contract() {
        // The public dispatcher (cached model or mock fallback) must always:
        // return one bit per token, never drop a locked token, and be
        // deterministic. With POLYMORPH_LAMR_MODEL unset (the cargo-test default)
        // this exercises the mock-fallback branch end to end.
        let token_ids: Vec<u32> = (0..64).collect();
        let lock_mask: Vec<bool> = (0..64).map(|i| i % 4 == 0).collect();
        let a = apply_lamr(&token_ids, &lock_mask);
        let b = apply_lamr(&token_ids, &lock_mask);
        assert_eq!(a.len(), token_ids.len());
        assert_eq!(a, b, "dispatcher must be deterministic");
        for (i, &locked) in lock_mask.iter().enumerate() {
            if locked {
                assert!(!a[i], "locked token {i} was dropped by apply_lamr");
            }
        }
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

        // A small token sequence with a mix of locked / unlocked positions.
        let token_ids: Vec<u32> = vec![2, 19690, 74292, 271, 2028, 2246, 6866, 311, 10368, 279];
        let lock_mask: Vec<bool> = vec![
            true, false, false, true, false, false, true, false, false, true,
        ];

        let mask_a =
            apply_lamr_with_model(&token_ids, &lock_mask, &model).expect("apply onnx a");
        let mask_b =
            apply_lamr_with_model(&token_ids, &lock_mask, &model).expect("apply onnx b");

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
    fn onnx_transitions_json_loads() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx transitions test: {} not found", path.display());
            return;
        }
        let tpath = path.parent().unwrap().join("transitions.json");
        let t = Transitions::from_json_path(&tpath).expect("load transitions.json");
        // Shapes validated by from_json_path; sanity-check the tag dimension.
        assert_eq!(t.trans.len(), NUM_TAGS);
        assert_eq!(t.start.len(), NUM_TAGS);
        assert_eq!(t.end.len(), NUM_TAGS);
    }

    #[test]
    fn onnx_all_unlocked_returns_full_length() {
        let path = smoke_model_path();
        if !path.exists() {
            eprintln!("skipping onnx full-length test: {} not found", path.display());
            return;
        }
        let model = LamrOnnx::load(&path).expect("load smoke model");
        let token_ids: Vec<u32> = vec![100, 200, 300, 400, 500];
        let lock_mask: Vec<bool> = vec![false; token_ids.len()];
        let mask = apply_lamr_with_model(&token_ids, &lock_mask, &model).expect("apply");
        assert_eq!(mask.len(), token_ids.len());
    }

    #[test]
    fn onnx_forward_drop_bits_rejects_oversize_sequence() {
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
            model.forward_drop_bits(&huge).is_err(),
            "oversize sequence must be rejected by the length guard"
        );
    }
}
