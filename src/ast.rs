use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::Mutex;

use tree_sitter::{wasmtime::Engine, Language as TsLanguage, Parser, TreeCursor, WasmStore};

use crate::sweep::sort_and_merge;
use crate::Language;

/// Shared wasmtime engine reused for every grammar instance. The store is per-
/// parse because tree-sitter consumes it via `Parser::set_wasm_store`.
fn engine() -> &'static Engine {
    static ENGINE: OnceCell<Engine> = OnceCell::new();
    ENGINE.get_or_init(Engine::default)
}

/// Cached WASM grammar bytes, read once from disk.
fn grammar_bytes(language: Language, grammars_dir: &Path) -> Result<&'static [u8]> {
    static JSON_BYTES: OnceCell<Vec<u8>> = OnceCell::new();
    static PY_BYTES: OnceCell<Vec<u8>> = OnceCell::new();

    let cache = match language {
        Language::Json => &JSON_BYTES,
        Language::Python => &PY_BYTES,
    };

    let bytes = cache.get_or_try_init(|| {
        let path = grammars_dir.join(language.grammar_filename());
        std::fs::read(&path).map_err(|e| {
            anyhow!(
                "failed to read grammar {}: {e}",
                path.display()
            )
        })
    })?;

    Ok(bytes.as_slice())
}

/// One-shot parse: build a fresh WasmStore, load the grammar, parse the input,
/// and walk the tree to collect structural byte intervals.
pub fn extract_ast_intervals(
    text: &str,
    language: Language,
    grammars_dir: &Path,
) -> Result<Vec<(usize, usize)>> {
    let bytes = grammar_bytes(language, grammars_dir)?;

    // wasmtime Engine isn't thread-safe across the same Store, so serialize
    // parser construction. Parsing itself is short.
    static LOCK: OnceCell<Mutex<()>> = OnceCell::new();
    let lock = LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap();

    let mut store = WasmStore::new(engine())
        .map_err(|e| anyhow!("WasmStore::new failed: {e}"))?;
    let ts_lang: TsLanguage = store
        .load_language(language.name(), bytes)
        .map_err(|e| anyhow!("load_language({}) failed: {e}", language.name()))?;

    let mut parser = Parser::new();
    parser
        .set_wasm_store(store)
        .map_err(|e| anyhow!("set_wasm_store failed: {e}"))?;
    parser
        .set_language(&ts_lang)
        .map_err(|e| anyhow!("set_language failed: {e}"))?;

    let tree = parser
        .parse(text, None)
        .ok_or_else(|| anyhow!("parser returned no tree"))?;

    let mut intervals = Vec::new();
    let mut cursor = tree.walk();
    collect(&mut cursor, language, &mut intervals);

    Ok(sort_and_merge(intervals))
}

fn collect(cursor: &mut TreeCursor, lang: Language, out: &mut Vec<(usize, usize)>) {
    loop {
        let node = cursor.node();
        let kind = node.kind();

        let is_structural = if !node.is_named() {
            // Anonymous nodes = punctuation + literal keywords (e.g. `{`, `def`).
            true
        } else {
            match lang {
                Language::Json => matches!(kind, "true" | "false" | "null" | "number"),
                Language::Python => {
                    matches!(kind, "none" | "true" | "false" | "integer" | "float")
                }
            }
        };

        if is_structural {
            out.push((node.start_byte(), node.end_byte()));
        }

        if cursor.goto_first_child() {
            collect(cursor, lang, out);
            cursor.goto_parent();
        }
        if !cursor.goto_next_sibling() {
            break;
        }
    }
}
