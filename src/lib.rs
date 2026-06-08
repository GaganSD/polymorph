pub mod adapters_common;
pub mod align;
pub mod ast;
pub mod bench;
pub mod ccr;
pub mod cli;
pub mod compress;
pub mod daac;
pub mod db;
pub mod dedup;
pub mod eval_metrics;
pub mod demo;
pub mod io_guard;
pub mod lamr;
pub mod label_ceiling;
pub mod lcm;
pub mod locking;
pub mod loghub;
pub mod mcp;
pub mod methods;
pub mod modernbert;
pub mod normalize;
pub mod selftest;
pub mod spandecode;
pub mod stats;
pub mod structural;
pub mod survival;
pub mod tokenizer;
pub mod transport;
pub mod triples;

pub use ast::extract_ast_intervals;
pub use daac::DaacScanner;
pub use locking::build_mask;
pub use tokenizer::token_spans;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Json,
    Python,
    /// No grammar — AST structural locking is skipped entirely. The correct mode
    /// for raw/free-text logs: forcing `Json`/`Python` on text that doesn't parse
    /// makes tree-sitter emit huge ERROR-node intervals that spuriously lock ~half
    /// the doc, crowding the pruner's drop budget and evicting free-text needles
    /// (the README's "structural floor hurts on prose"). Structure is then locked
    /// only by DAAC keyword matches; the model handles the prose.
    PlainText,
}

impl Language {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "json" => Some(Language::Json),
            "python" => Some(Language::Python),
            "text" | "plain" | "plaintext" | "log" => Some(Language::PlainText),
            _ => None,
        }
    }

    /// Grammar WASM filename. Never called for [`Language::PlainText`] —
    /// `extract_ast_intervals` short-circuits it before any grammar load.
    pub fn grammar_filename(self) -> &'static str {
        match self {
            Language::Json => "tree-sitter-json.wasm",
            Language::Python => "tree-sitter-python.wasm",
            Language::PlainText => "",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Language::Json => "json",
            Language::Python => "python",
            Language::PlainText => "text",
        }
    }
}

pub struct LockResult {
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
    pub ast_intervals: Vec<(usize, usize)>,
    pub daac_token_intervals: Vec<(usize, usize)>,
    /// M1: per-token lock mask. `true` = structural/keyword, must not be dropped.
    pub mask: Vec<bool>,
    /// M2: LaMR drop mask. `true` = drop. Always `false` where `mask[i]` is
    /// `true`. Produced by the ONNX-backed pruner when an exported model is
    /// available (env `POLYMORPH_LAMR_MODEL` or the default artifact path),
    /// otherwise by the deterministic mock fallback.
    pub drop_mask: Vec<bool>,
    /// Convenience: count of tokens kept after the drop mask is applied.
    pub kept_tokens: usize,
}

/// Resolves the directory holding `tree-sitter-*.wasm` files. Honors the
/// `POLYMORPH_GRAMMARS_DIR` env var, otherwise walks up from the binary
/// location (`target/debug/polymorph-mcp` → repo root → `grammars/`), and
/// finally falls back to a relative `grammars` path.
pub fn resolve_grammars_dir() -> std::path::PathBuf {
    if let Ok(env) = std::env::var("POLYMORPH_GRAMMARS_DIR") {
        return std::path::PathBuf::from(env);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(candidate) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("grammars"))
        {
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from("grammars")
}

/// Shared structural-locking core: tokenize, scan keywords (DAAC), extract AST
/// intervals, and intersect them into the per-token lock mask. Both
/// `lock_payload` (mock pruner on) and `compress_deterministic` (pruner off)
/// build on this, so the masking logic lives in one place.
fn lock_core(
    text: &str,
    language: Language,
    keywords: &[String],
    grammars_dir: &std::path::Path,
) -> anyhow::Result<(
    Vec<u32>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<bool>,
)> {
    let (token_ids, token_spans) = tokenizer::token_spans(text)?;
    let mut scanner = daac::DaacScanner::build(keywords)?;
    let daac_token_intervals = scanner.scan(&token_ids);
    let ast_intervals = ast::extract_ast_intervals(text, language, grammars_dir)?;
    let mask = locking::build_mask(
        token_spans.len(),
        &token_spans,
        &ast_intervals,
        &daac_token_intervals,
    );
    Ok((
        token_ids,
        token_spans,
        ast_intervals,
        daac_token_intervals,
        mask,
    ))
}

pub fn lock_payload(
    text: &str,
    language: Language,
    keywords: &[String],
    grammars_dir: &std::path::Path,
) -> anyhow::Result<LockResult> {
    let (token_ids, token_spans, ast_intervals, daac_token_intervals, mask) =
        lock_core(text, language, keywords, grammars_dir)?;
    let drop_mask = lamr::apply_lamr(&token_ids, &mask, &token_spans, text);
    let kept_tokens = drop_mask.iter().filter(|&&d| !d).count();
    Ok(LockResult {
        token_ids,
        token_spans,
        ast_intervals,
        daac_token_intervals,
        mask,
        drop_mask,
        kept_tokens,
    })
}

/// Deterministic-only locking with the pruner OFF (Identity). `drop_mask` is
/// all-`false`, so the only compression reflected here is structural locking
/// (plus whatever upstream dedup/CCR already removed) — never the random mock
/// drops. This is what the benchmark and the deterministic baseline measure;
/// the full pluggable pruner seam (Identity/Mock/ONNX) is deferred (see TODOS.md).
pub fn compress_deterministic(
    text: &str,
    language: Language,
    keywords: &[String],
    grammars_dir: &std::path::Path,
) -> anyhow::Result<LockResult> {
    let (token_ids, token_spans, ast_intervals, daac_token_intervals, mask) =
        lock_core(text, language, keywords, grammars_dir)?;
    let kept_tokens = token_ids.len();
    let drop_mask = vec![false; mask.len()];
    Ok(LockResult {
        token_ids,
        token_spans,
        ast_intervals,
        daac_token_intervals,
        mask,
        drop_mask,
        kept_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_from_str_known() {
        assert_eq!(Language::from_str("json"), Some(Language::Json));
        assert_eq!(Language::from_str("python"), Some(Language::Python));
        assert_eq!(Language::from_str("typescript"), None);
    }

    #[test]
    fn language_name_and_grammar_filename_match() {
        assert_eq!(Language::Json.name(), "json");
        assert_eq!(Language::Python.name(), "python");
        assert_eq!(Language::Json.grammar_filename(), "tree-sitter-json.wasm");
        assert_eq!(Language::Python.grammar_filename(), "tree-sitter-python.wasm");
    }

    #[test]
    fn resolve_grammars_dir_honors_env_var() {
        let prev = std::env::var("POLYMORPH_GRAMMARS_DIR").ok();
        std::env::set_var("POLYMORPH_GRAMMARS_DIR", "/tmp/some/path");
        let resolved = resolve_grammars_dir();
        assert_eq!(resolved, std::path::PathBuf::from("/tmp/some/path"));
        match prev {
            Some(v) => std::env::set_var("POLYMORPH_GRAMMARS_DIR", v),
            None => std::env::remove_var("POLYMORPH_GRAMMARS_DIR"),
        }
    }

    #[test]
    fn lock_payload_returns_consistent_lengths() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");
        let res = lock_payload(r#"{"a":1}"#, Language::Json, &[], &dir).unwrap();
        assert_eq!(res.mask.len(), res.token_spans.len());
        assert_eq!(res.token_ids.len(), res.mask.len());
    }

    #[test]
    fn compress_deterministic_drops_nothing() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");
        let res = compress_deterministic(r#"{"a":1,"b":2}"#, Language::Json, &[], &dir).unwrap();
        assert!(res.drop_mask.iter().all(|&d| !d), "Identity pruner drops nothing");
        assert_eq!(res.kept_tokens, res.token_ids.len());
    }

    #[test]
    fn mask_invariant_ast_and_daac_lock_tokens() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");
        let text = r#"{"secret":"x"}"#;
        let res = lock_payload(text, Language::Json, &["secret".to_string()], &dir).unwrap();
        for (i, &locked) in res.mask.iter().enumerate() {
            let (s, e) = res.token_spans[i];
            let in_ast = res
                .ast_intervals
                .iter()
                .any(|&(a, b)| a < e && s < b);
            let in_daac = res
                .daac_token_intervals
                .iter()
                .any(|&(a, b)| a <= i && i < b);
            if in_ast || in_daac {
                assert!(locked, "token {i} must be locked");
            }
        }
    }
}
