//! Pure-Rust ModernBERT tokenizer (encode-with-spans + decode).
//!
//! The LaMR pruner shipped in the runtime (`mb_v0`) is trained on
//! **ModernBERT-base** token ids (vocab 50368), NOT cl100k. Feeding it cl100k ids
//! (what [`crate::tokenizer`] produces) yields garbage, so the LaMR stage must
//! tokenize in ModernBERT space. We do this without the HF `tokenizers` crate
//! (which pulls in the `onig` C library and breaks a clean `cargo install` — the
//! same reason `tract` was chosen over `ort`): ModernBERT's tokenizer is a
//! GPT-2-family **byte-level BPE**, which the pure-Rust `tiktoken-rs` engine can
//! reproduce exactly when its rank map is keyed by each token's *raw bytes*
//! (`unbytelevel(token) -> id`) and driven by the GPT-2 split regex. The id
//! ordering of merge-result tokens matches the merges-list order (verified), so
//! tiktoken's rank-based merge == HF's explicit-merges BPE.
//!
//! Two ModernBERT-specific wrinkles are handled around the BPE engine:
//!   * **Added tokens** — the multi-space runs (ids 50254–50276, *2 to 24*
//!     spaces — common in indented logs), the PII placeholders
//!     (`|||IP_ADDRESS|||` …), and the bracket specials (`[CLS]`, `[MASK]` …).
//!     HF extracts these from the text *before* BPE, longest-match first; we do
//!     the same in a pre-pass, then BPE the gaps. (tiktoken's own special-token
//!     path can't be reused here: it builds the alternation without length
//!     sorting, so it would not honour longest-match on overlapping space runs.)
//!   * **decode** — surviving ids are detokenised by concatenating each id's raw
//!     bytes (BPE token bytes or the added-token's literal bytes).
//!
//! Contract mirrors `ml_pipeline/polymorph_lamr/label/align.py`:
//! `encode_with_spans(text)` returns ids + per-token `(start_byte, end_byte)`
//! spans into `text`, and `decode(ids)` round-trips. Validated to byte-parity
//! against HF fixtures in `tests/fixtures/modernbert_parity.json`.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

/// The GPT-2 / ByteLevel pre-tokenizer split pattern (ModernBERT uses
/// `ByteLevel{use_regex:true}`). `fancy-regex` (tiktoken's engine) supports the
/// `(?!\S)` lookahead and `\p{L}`/`\p{N}` classes.
const GPT2_SPLIT_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Bundled ModernBERT tokenizer definition (HF `answerdotai/ModernBERT-base`).
/// Embedded so the runtime is self-contained — no path/env needed for the
/// tokenizer (the model weights are still loaded from `POLYMORPH_LAMR_MODEL`).
const TOKENIZER_JSON: &str = include_str!("../assets/modernbert/tokenizer.json");

/// GPT-2 byte↔unicode table: byte value (0..=255) → the printable char HF maps it
/// to inside vocab strings. Inverting it recovers a vocab token's raw bytes.
fn bytes_to_unicode() -> [char; 256] {
    // bs: the byte values that already map to a printable, non-space glyph and so
    // stay as themselves; everything else is shifted into the 256.. private range.
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).expect("valid scalar");
    }
    table
}

/// A loaded ModernBERT tokenizer: the BPE engine plus the added-token tables.
pub struct ModernBertTokenizer {
    bpe: CoreBPE,
    /// id → raw bytes, for every id we can emit (BPE tokens *and* added tokens).
    id_to_bytes: HashMap<u32, Vec<u8>>,
    /// Added tokens as (content_bytes, id), sorted by content length DESC so a
    /// left-to-right scan honours longest-match (24-space run before 2-space).
    added_sorted: Vec<(Vec<u8>, u32)>,
}

impl ModernBertTokenizer {
    /// Parse the embedded tokenizer.json and build the BPE engine + added-token
    /// tables. Done once (cached via [`get`]).
    fn build() -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(TOKENIZER_JSON)
            .context("parse bundled ModernBERT tokenizer.json")?;
        let model = v
            .get("model")
            .ok_or_else(|| anyhow!("tokenizer.json: no model"))?;
        let vocab = model
            .get("vocab")
            .and_then(|x| x.as_object())
            .ok_or_else(|| anyhow!("tokenizer.json: no model.vocab object"))?;
        let added = v
            .get("added_tokens")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("tokenizer.json: no added_tokens array"))?;

        // Char → byte inverse of the GPT-2 byte-level map.
        let b2u = bytes_to_unicode();
        let mut u2b: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &c) in b2u.iter().enumerate() {
            u2b.insert(c, b as u8);
        }

        // Added-token ids are handled by the pre-pass, never by BPE — exclude
        // them from the BPE rank map so the engine can't emit or merge into one.
        let mut added_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut added_sorted: Vec<(Vec<u8>, u32)> = Vec::with_capacity(added.len());
        let mut id_to_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
        for t in added {
            let id = t
                .get("id")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow!("added token: no id"))? as u32;
            let content = t
                .get("content")
                .and_then(|x| x.as_str())
                .ok_or_else(|| anyhow!("added token {id}: no content"))?;
            added_ids.insert(id);
            let bytes = content.as_bytes().to_vec();
            added_sorted.push((bytes.clone(), id));
            id_to_bytes.insert(id, bytes);
        }
        // Longest content first; tie-break by id for determinism.
        added_sorted.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        // BPE rank map: unbytelevel(token_string) -> id, for non-added vocab.
        // FxHashMap to match tiktoken-rs's `CoreBPE::new` signature exactly.
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.reserve(vocab.len());
        for (tok, id_val) in vocab {
            let id = id_val
                .as_u64()
                .ok_or_else(|| anyhow!("vocab {tok}: non-int id"))? as u32;
            if added_ids.contains(&id) {
                continue;
            }
            let mut raw = Vec::with_capacity(tok.len());
            for ch in tok.chars() {
                let b = *u2b.get(&ch).ok_or_else(|| {
                    anyhow!("vocab token {tok:?}: char {ch:?} not in byte-level map")
                })?;
                raw.push(b);
            }
            encoder.insert(raw.clone(), id);
            id_to_bytes.insert(id, raw);
        }

        // CoreBPE needs the rank map keyed by raw bytes + the split pattern; we
        // drive added tokens ourselves, so pass an empty special set.
        let bpe = CoreBPE::new(encoder, FxHashMap::default(), GPT2_SPLIT_PATTERN)
            .map_err(|e| anyhow!("building ModernBERT CoreBPE: {e}"))?;

        Ok(ModernBertTokenizer {
            bpe,
            id_to_bytes,
            added_sorted,
        })
    }

    /// Encode `text` to ModernBERT ids with `(start_byte, end_byte)` spans into
    /// `text`. Mirrors `encode_with_spans(text, "modernbert")`: no special tokens
    /// are prepended; added tokens (space runs / PII / brackets) are extracted
    /// longest-match-first; the gaps are byte-level BPE'd. Spans are byte ranges;
    /// `text.as_bytes()[start..end]` is the token's surface bytes.
    pub fn encode_with_spans(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut ids: Vec<u32> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();

        let mut pos = 0usize; // byte cursor
        let mut run_start = 0usize; // start of the current plain (non-added) run
        while pos < n {
            if let Some((content, id)) = self.match_added_at(bytes, pos) {
                // Flush the plain run before this added token.
                if run_start < pos {
                    self.bpe_run_with_spans(text, run_start, pos, &mut ids, &mut spans);
                }
                ids.push(*id);
                spans.push((pos, pos + content.len()));
                pos += content.len();
                run_start = pos;
            } else {
                pos += 1;
            }
        }
        if run_start < n {
            self.bpe_run_with_spans(text, run_start, n, &mut ids, &mut spans);
        }
        (ids, spans)
    }

    /// Longest added-token whose content matches `bytes` starting at `pos`.
    fn match_added_at<'a>(&'a self, bytes: &[u8], pos: usize) -> Option<&'a (Vec<u8>, u32)> {
        self.added_sorted.iter().find(|(content, _)| {
            let end = pos + content.len();
            end <= bytes.len() && &bytes[pos..end] == content.as_slice()
        })
    }

    /// BPE the byte range `text[start..end]` and push ids + spans (offset into
    /// `text`). cl100k-style byte-span reconstruction: `_decode_native_and_split`
    /// yields each token's raw bytes, whose lengths tile the run contiguously.
    fn bpe_run_with_spans(
        &self,
        text: &str,
        start: usize,
        end: usize,
        ids: &mut Vec<u32>,
        spans: &mut Vec<(usize, usize)>,
    ) {
        let run = &text[start..end];
        let run_ids = self.bpe.encode_ordinary(run);
        let mut cursor = start;
        for (tid, tok_bytes) in run_ids
            .iter()
            .copied()
            .zip(self.bpe._decode_native_and_split(run_ids.clone()))
        {
            let s = cursor;
            cursor += tok_bytes.len();
            ids.push(tid);
            spans.push((s, cursor));
        }
    }

    /// Detokenise `ids` to text by concatenating each id's raw bytes. Mirrors
    /// `decode_tokens(ids, "modernbert")`.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out: Vec<u8> = Vec::new();
        for id in ids {
            let b = self
                .id_to_bytes
                .get(id)
                .ok_or_else(|| anyhow!("decode: unknown ModernBERT id {id}"))?;
            out.extend_from_slice(b);
        }
        String::from_utf8(out).map_err(|e| anyhow!("ModernBERT decode produced invalid UTF-8: {e}"))
    }
}

static TOKENIZER: OnceLock<Result<ModernBertTokenizer>> = OnceLock::new();

/// Process-wide cached tokenizer. The parse + rank-map build is non-trivial
/// (~50k vocab entries) so it happens once. Returns a reference or the build error.
pub fn get() -> Result<&'static ModernBertTokenizer> {
    match TOKENIZER.get_or_init(|| ModernBertTokenizer::build()) {
        Ok(t) => Ok(t),
        Err(e) => Err(anyhow!("ModernBERT tokenizer unavailable: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures() -> Vec<serde_json::Value> {
        let p =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/modernbert_parity.json");
        let raw = std::fs::read_to_string(p).expect("read parity fixtures");
        serde_json::from_str(&raw).expect("parse parity fixtures")
    }

    #[test]
    fn byte_level_map_covers_all_256() {
        let b2u = bytes_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for c in b2u {
            assert!(seen.insert(c), "byte-level map must be injective");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn encoder_covers_all_valid_utf8_bytes() {
        // Every byte that can appear in valid UTF-8 must have a single-byte token,
        // or BPE of some input would fail. The 13 bytes ModernBERT's byte-level
        // vocab omits (0xC0, 0xC1, 0xF5..=0xFF) are exactly the bytes that CANNOT
        // occur in valid UTF-8 — and our input is always a Rust `&str` — so their
        // absence is correct, not a gap.
        const INVALID_UTF8: [u8; 13] = [
            0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
        ];
        let tok = get().expect("build tokenizer");
        for b in 0u8..=255 {
            if INVALID_UTF8.contains(&b) {
                continue;
            }
            assert!(
                tok.id_to_bytes.values().any(|v| v.as_slice() == [b]),
                "no ModernBERT token for valid-UTF-8 byte {b}"
            );
        }
    }

    #[test]
    fn unicode_stress_round_trips() {
        // Exercises 2/3/4-byte UTF-8 sequences end-to-end: every byte they decompose
        // into must be BPE-encodable and decode back exactly.
        let tok = get().expect("build tokenizer");
        let s = "café ünïcødé 日本語 emoji 🌍🚀 math ∑∫ Ω → ✓ — \u{00C0}\u{00FF}";
        let (ids, spans) = tok.encode_with_spans(s);
        assert_eq!(ids.len(), spans.len());
        assert_eq!(tok.decode(&ids).unwrap(), s, "unicode stress round-trip");
    }

    #[test]
    fn parity_ids_and_spans_match_hf() {
        let tok = get().expect("build tokenizer");
        for fx in fixtures() {
            let text = fx["text"].as_str().unwrap();
            let want_ids: Vec<u32> = fx["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let want_spans: Vec<(usize, usize)> = fx["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let a = s.as_array().unwrap();
                    (
                        a[0].as_u64().unwrap() as usize,
                        a[1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            let (ids, spans) = tok.encode_with_spans(text);
            assert_eq!(ids, want_ids, "id mismatch for {text:?}");
            assert_eq!(spans, want_spans, "span mismatch for {text:?}");
        }
    }

    #[test]
    fn decode_round_trips_fixtures() {
        let tok = get().expect("build tokenizer");
        for fx in fixtures() {
            let text = fx["text"].as_str().unwrap();
            let (ids, _) = tok.encode_with_spans(text);
            assert_eq!(
                tok.decode(&ids).unwrap(),
                text,
                "decode round-trip for {text:?}"
            );
        }
    }

    #[test]
    fn spans_slice_back_to_surface_bytes() {
        let tok = get().expect("build tokenizer");
        for fx in fixtures() {
            let text = fx["text"].as_str().unwrap();
            let raw = text.as_bytes();
            let (ids, spans) = tok.encode_with_spans(text);
            assert_eq!(ids.len(), spans.len());
            for &(s, e) in &spans {
                assert!(
                    s < e && e <= raw.len(),
                    "span ({s},{e}) out of range for {text:?}"
                );
            }
        }
    }
}
