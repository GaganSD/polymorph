pub mod ast;
pub mod daac;
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
