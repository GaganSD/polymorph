use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use tree_sitter::{
    wasmtime::{Config, Engine},
    Language as TsLanguage, Parser, TreeCursor, WasmStore,
};

use crate::locking::sort_and_merge;
use crate::Language;

/// Trusted local grammar WASM files; cap prevents unbounded disk read if path is poisoned.
const MAX_GRAMMAR_BYTES: u64 = 10 * 1024 * 1024;

/// Hard wall-clock budget for WASM Tree-sitter parse (algorithmic complexity DoS guard).
const PARSE_TIMEOUT: Duration = Duration::from_millis(50);

/// Wasmtime stack budget per grammar invocation (bytes).
const MAX_WASM_STACK: usize = 256 * 1024;

/// Reserved linear memory for WASM store (bytes).
const WASM_MEMORY_RESERVATION: u64 = 16 * 1024 * 1024;

static RUNTIME: OnceCell<tokio::runtime::Runtime> = OnceCell::new();
static ENGINE: OnceCell<Engine> = OnceCell::new();

fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .worker_threads(2)
            .thread_name("polymorph-wasm-worker")
            .build()
            .expect("failed to build tokio runtime for WASM sandbox")
    })
}

/// Sandboxed wasmtime engine: tight stack and memory reservation.
///
/// Tree-sitter owns the WasmStore and does not expose `Store::add_fuel`, so fuel
/// metering is enforced externally via `PARSE_TIMEOUT` + `spawn_blocking`.
fn engine() -> &'static Engine {
    ENGINE.get_or_init(|| {
        let mut config = Config::new();
        config.max_wasm_stack(MAX_WASM_STACK);
        config.memory_reservation(WASM_MEMORY_RESERVATION);
        config.memory_reservation_for_growth(1024 * 1024);
        // Tree-sitter WasmStore does not expose Store::add_fuel; wall-clock timeout
        // in `extract_ast_intervals` enforces execution bounds instead of fuel.
        Engine::new(&config).expect("failed to create sandboxed wasmtime Engine")
    })
}

fn read_bounded_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|e| anyhow!("failed to open {}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| anyhow!("metadata: {e}"))?.len();
    if len > max_bytes {
        return Err(anyhow!(
            "file {} is {len} bytes, exceeds cap {max_bytes}",
            path.display()
        ));
    }
    let mut limited = file.take(max_bytes);
    let mut bytes = Vec::with_capacity(len as usize);
    limited
        .read_to_end(&mut bytes)
        .map_err(|e| anyhow!("read {}: {e}", path.display()))?;
    Ok(bytes)
}

/// Cached WASM grammar bytes, read once from disk under a byte cap.
fn grammar_bytes(language: Language, grammars_dir: &Path) -> Result<&'static [u8]> {
    static JSON_BYTES: OnceCell<Vec<u8>> = OnceCell::new();
    static PY_BYTES: OnceCell<Vec<u8>> = OnceCell::new();

    let cache = match language {
        Language::Json => &JSON_BYTES,
        Language::Python => &PY_BYTES,
    };

    let bytes = cache.get_or_try_init(|| {
        let path = grammars_dir.join(language.grammar_filename());
        read_bounded_file(&path, MAX_GRAMMAR_BYTES)
    })?;

    Ok(bytes.as_slice())
}

/// Reusable WASM-backed parser — avoids re-instantiating WasmStore on every call.
struct CachedWasmParser {
    parser: Parser,
}

static JSON_PARSER: OnceCell<Mutex<CachedWasmParser>> = OnceCell::new();
static PY_PARSER: OnceCell<Mutex<CachedWasmParser>> = OnceCell::new();

fn parser_cache(language: Language) -> &'static OnceCell<Mutex<CachedWasmParser>> {
    match language {
        Language::Json => &JSON_PARSER,
        Language::Python => &PY_PARSER,
    }
}

fn init_wasm_parser(language: Language, grammars_dir: &Path) -> Result<CachedWasmParser> {
    let bytes = grammar_bytes(language, grammars_dir)?;
    let mut store = WasmStore::new(engine()).map_err(|e| anyhow!("WasmStore::new failed: {e}"))?;
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

    Ok(CachedWasmParser { parser })
}

fn cached_parser(language: Language, grammars_dir: &Path) -> Result<&'static Mutex<CachedWasmParser>> {
    parser_cache(language).get_or_try_init(|| init_wasm_parser(language, grammars_dir).map(Mutex::new))
}

/// Parse off the MCP stdio thread with a 50ms timeout on the hot path.
pub fn extract_ast_intervals(
    text: &str,
    language: Language,
    grammars_dir: &Path,
) -> Result<Vec<(usize, usize)>> {
    // Ensure WASM grammar is compiled before the timed window starts.
    cached_parser(language, grammars_dir)?;

    let text = text.to_owned();
    let grammars_dir = grammars_dir.to_path_buf();

    runtime().block_on(async {
        tokio::time::timeout(
            PARSE_TIMEOUT,
            tokio::task::spawn_blocking(move || {
                extract_ast_intervals_blocking(&text, language, &grammars_dir)
            }),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "AST parse exceeded {}ms — refusing (complexity DoS guard)",
                PARSE_TIMEOUT.as_millis()
            )
        })?
        .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
    })
}

fn extract_ast_intervals_blocking(
    text: &str,
    language: Language,
    grammars_dir: &Path,
) -> Result<Vec<(usize, usize)>> {
    let mutex = cached_parser(language, grammars_dir)?;
    let mut guard = mutex.lock().unwrap();

    let tree = guard
        .parser
        .parse(text, None)
        .ok_or_else(|| anyhow!("parser returned no tree"))?;

    let mut intervals = Vec::new();
    let mut cursor = tree.walk();
    collect_iterative(&mut cursor, language, &mut intervals);

    Ok(sort_and_merge(intervals))
}

fn collect_iterative(cursor: &mut TreeCursor, lang: Language, out: &mut Vec<(usize, usize)>) {
    loop {
        let node = cursor.node();
        let kind = node.kind();
        let is_structural = if !node.is_named() {
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
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}
