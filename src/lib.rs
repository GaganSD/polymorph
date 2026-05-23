pub mod ast;
pub mod daac;
pub mod db;
pub mod io_guard;
pub mod mcp;
pub mod selftest;
pub mod sweep;
pub mod tokens;

pub use ast::extract_ast_intervals;
pub use daac::DaacScanner;
pub use sweep::build_mask;
pub use tokens::token_spans;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Json,
    Python,
}

impl Language {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "json" => Some(Language::Json),
            "python" => Some(Language::Python),
            _ => None,
        }
    }

    pub fn grammar_filename(self) -> &'static str {
        match self {
            Language::Json => "tree-sitter-json.wasm",
            Language::Python => "tree-sitter-python.wasm",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Language::Json => "json",
            Language::Python => "python",
        }
    }
}

pub struct LockResult {
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
    pub ast_intervals: Vec<(usize, usize)>,
    pub daac_token_intervals: Vec<(usize, usize)>,
    pub mask: Vec<bool>,
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

pub fn lock_payload(
    text: &str,
    language: Language,
    keywords: &[String],
    grammars_dir: &std::path::Path,
) -> anyhow::Result<LockResult> {
    let (token_ids, token_spans) = tokens::token_spans(text)?;
    let scanner = daac::DaacScanner::build(keywords)?;
    let daac_token_intervals = scanner.scan(&token_ids);
    let ast_intervals = ast::extract_ast_intervals(text, language, grammars_dir)?;
    let mask = sweep::build_mask(
        token_spans.len(),
        &token_spans,
        &ast_intervals,
        &daac_token_intervals,
    );
    Ok(LockResult {
        token_ids,
        token_spans,
        ast_intervals,
        daac_token_intervals,
        mask,
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
        // SAFETY: tests run sequentially within a module by default; the env
        // mutation here is scoped and restored.
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

}
