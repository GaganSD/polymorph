//! MCP server implementation using the official `rmcp` SDK.
//!
//! All wire framing, JSON-RPC handling, schema publication, and capability
//! advertisement are delegated to rmcp. Each tool is a `#[tool]`-annotated
//! async method on `PolymorphServer`. The core AST/DAAC/SQLite logic is
//! untouched — this module is only the protocol routing wrapper.

use std::path::PathBuf;
use std::sync::Arc;

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, ErrorData, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
    },
    tool, tool_handler, tool_router, ServerHandler,
};
use serde_json::{json, Value};

use crate::ccr::{self, CacheMiss, CcrOpts};
use crate::db::DbHandle;
use crate::dedup::{self, DedupOpts};
use crate::io_guard::{
    check_compress_array_input, check_compress_log_input, check_lcm_append_input,
    check_lcm_node_input, check_lock_mask_input, check_retrieve_cache_input, CompressArrayInput,
    CompressLogInput, LcmAppendInput, LcmNodeInput, LockMaskInput, RetrieveCacheInput,
    MAX_LOG_FILE_BYTES, MAX_MASK_TOKENS,
};
use crate::lcm::{self, NotFound as LcmNotFound, DEFAULT_SOFT_THRESHOLD};
use crate::{compress, lock_payload, tokenizer, Language};

/// Upper bound on ModernBERT tokens the LaMR pruner scores per `compress_log`
/// call (after dedup). Beyond it the tail is kept verbatim — a latency guard for
/// pathological all-unique prose; a kept tail never loses a needle.
const COMPRESS_LOG_NEURAL_CAP: usize = 65_536;

/// Shared server state. Cheap to clone — `DbHandle` is an actor handle, the
/// grammars path is `Arc`-wrapped for the same reason.
#[derive(Clone)]
pub struct PolymorphServer {
    pub db: DbHandle,
    pub grammars_dir: Arc<PathBuf>,
    #[allow(dead_code)]
    tool_router: ToolRouter<PolymorphServer>,
}

impl PolymorphServer {
    pub fn new(db: DbHandle, grammars_dir: PathBuf) -> Self {
        Self {
            db,
            grammars_dir: Arc::new(grammars_dir),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl PolymorphServer {
    #[tool(
        description = "Produce a per-token lock+drop mask. lock_mask[i]=true means structural/keyword (must keep); drop_mask[i]=true means the mock LaMR pruner suggests dropping this unlocked token."
    )]
    async fn lock_mask(
        &self,
        Parameters(input): Parameters<LockMaskInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_lock_mask_input(&input).map_err(validation_err)?;
        let lang = Language::from_str(&input.language).ok_or_else(|| {
            let language = input.language.clone();
            ErrorData::invalid_params(
                format!("unsupported language: {language}"),
                Some(json!({
                    "error": "unsupported_language",
                    "language": language,
                    "hint": "use one of: json, python",
                })),
            )
        })?;
        let grammars = self.grammars_dir.clone();
        let res = tokio::task::spawn_blocking(move || {
            lock_payload(&input.text, lang, &input.keywords, grammars.as_path())
        })
        .await
        .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?
        .map_err(|e| internal_err("lock_payload_failed", format!("lock_payload failed: {e}")))?;
        if res.mask.len() > MAX_MASK_TOKENS {
            return Err(ErrorData::invalid_params(
                format!("lock_mask output exceeds max token count ({MAX_MASK_TOKENS}) \u{2014} reduce input text"),
                Some(json!({
                    "error": "output_too_large",
                    "max_tokens": MAX_MASK_TOKENS,
                    "hint": "reduce input text",
                })),
            ));
        }
        Ok(CallToolResult::structured(json!({
            "tokens": res.token_ids.len(),
            "kept_tokens": res.kept_tokens,
            "mask": res.mask,
            "drop_mask": res.drop_mask,
        })))
    }

    #[tool(
        description = "Compress a log/trace block into a smaller payload that still contains the answer. Deterministic template/run-length dedup + structural token-locking, then the trained LaMR pruner drops redundant prose to a target rate. Structural fields, error codes and trace ids are preserved; the full original is cached and retrievable via polymorph_retrieve_cache (cache_id). Accepts inline `text` or a local file `path` (use `path` for large logs). Reason over the returned `compressed` text."
    )]
    async fn compress_log(
        &self,
        Parameters(input): Parameters<CompressLogInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_compress_log_input(&input).map_err(validation_err)?;
        let lang = input
            .language
            .as_deref()
            .and_then(Language::from_str)
            .unwrap_or(Language::PlainText);
        let original = match (input.text.clone(), input.path.clone()) {
            (Some(t), _) => t,
            (None, Some(p)) => {
                read_log_file(&p).map_err(|e| internal_err("read_failed", e.to_string()))?
            }
            _ => unreachable!("validated: exactly one of text/path"),
        };
        let keywords = input.keywords.clone();
        let target_rate = input.target_rate;
        let grammars = self.grammars_dir.clone();
        let db = self.db.clone();

        let out = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
            let input_tokens = tokenizer::count_tokens(&original)?;
            // Deterministic template/run-length dedup first — bounds the neural
            // workload so a repetitive 10 MB log doesn't fan out to thousands of
            // windows.
            let plan = dedup::dedup_plan(&original, DedupOpts::default());
            let res = compress::compress_text(
                &plan.reduced,
                lang,
                &keywords,
                grammars.as_path(),
                target_rate,
                Some(COMPRESS_LOG_NEURAL_CAP),
            )?;
            // Cache the ORIGINAL for reversibility (retrievable via cache_id).
            let cache_id = ccr::stash(&Value::String(original), &db)?;
            let ratio = input_tokens as f64 / res.output_tokens.max(1) as f64;
            Ok(json!({
                "compressed": res.compressed,
                "cache_id": cache_id,
                "input_tokens": input_tokens,
                "output_tokens": res.output_tokens,
                "ratio": ratio,
                "dedup_elided_lines": plan.elided_line_count(),
                "used_model": res.used_model,
            }))
        })
        .await
        .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?
        .map_err(|e| internal_err("compress_log_failed", format!("compress_log failed: {e}")))?;
        Ok(CallToolResult::structured(out))
    }

    #[tool(
        description = "Compress a large JSON array by keeping head + tail elements and stashing the middle in the local SQLite cache. Returns the compressed array and a cache_id for retrieval."
    )]
    async fn compress_array(
        &self,
        Parameters(input): Parameters<CompressArrayInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_compress_array_input(&input).map_err(validation_err)?;
        let mut opts = CcrOpts::default();
        if let Some(h) = input.head_keep {
            opts.head_keep = h;
        }
        if let Some(t) = input.tail_keep {
            opts.tail_keep = t;
        }
        let db = self.db.clone();
        let value = input.value;
        let cache = input.cache;
        let res = tokio::task::spawn_blocking(move || ccr::compress_array(value, opts, &db, cache))
            .await
            .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?
            .map_err(|e| {
                internal_err(
                    "compress_array_failed",
                    format!("compress_array failed: {e}"),
                )
            })?;
        Ok(CallToolResult::structured(json!({
            "compressed": res.compressed,
            "cache_id": res.cache_id,
            "omitted_count": res.omitted_count,
        })))
    }

    #[tool(
        description = "Retrieve the original middle slice of a previously-compressed JSON array by its cache_id."
    )]
    async fn polymorph_retrieve_cache(
        &self,
        Parameters(input): Parameters<RetrieveCacheInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_retrieve_cache_input(&input).map_err(validation_err)?;
        let db = self.db.clone();
        let cache_id = input.cache_id.clone();
        let res = tokio::task::spawn_blocking(move || ccr::retrieve(&input.cache_id, &db))
            .await
            .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?;
        match res {
            Ok(value) => Ok(CallToolResult::structured(json!({"value": value}))),
            Err(e) if e.is::<CacheMiss>() => Err(ErrorData::invalid_params(
                "cache_miss",
                Some(json!({
                    "error": "cache_miss",
                    "cache_id": cache_id,
                    "hint": "cache entry expired or never existed",
                })),
            )),
            Err(e) => Err(internal_err(
                "retrieve_failed",
                format!("retrieve failed: {e}"),
            )),
        }
    }

    #[tool(
        description = "Append a conversational turn to the LCM store. If the conversation's active token count exceeds soft_threshold (default 80000), the oldest turns are archived into a Depth-0 summary node."
    )]
    async fn lcm_append(
        &self,
        Parameters(input): Parameters<LcmAppendInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_lcm_append_input(&input).map_err(validation_err)?;
        let threshold = input.soft_threshold.unwrap_or(DEFAULT_SOFT_THRESHOLD);
        let db = self.db.clone();
        let res = tokio::task::spawn_blocking(move || {
            lcm::append_and_maybe_archive(
                &input.conversation_id,
                &input.role,
                &input.content,
                threshold,
                &db,
            )
        })
        .await
        .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?
        .map_err(|e| internal_err("lcm_append_failed", format!("lcm_append failed: {e}")))?;
        Ok(CallToolResult::structured(json!({
            "turn_id": res.turn_id,
            "turn_index": res.turn_index,
            "tokens": res.tokens,
            "archived_node_id": res.archived_node_id,
        })))
    }

    #[tool(
        description = "Return metadata about an archived summary node (child_count, total_tokens, roles, created_at)."
    )]
    async fn lcm_describe(
        &self,
        Parameters(input): Parameters<LcmNodeInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_lcm_node_input(&input).map_err(validation_err)?;
        let db = self.db.clone();
        let node_id = input.node_id.clone();
        let res = tokio::task::spawn_blocking(move || lcm::describe(&input.node_id, &db))
            .await
            .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?;
        match res {
            Ok(meta) => Ok(CallToolResult::structured(
                serde_json::to_value(&meta).unwrap_or(Value::Null),
            )),
            Err(e) if e.is::<LcmNotFound>() => Err(lcm_not_found(&node_id)),
            Err(e) => Err(internal_err(
                "lcm_describe_failed",
                format!("lcm_describe failed: {e}"),
            )),
        }
    }

    #[tool(
        description = "Return the original verbatim turns archived under a summary node, in turn-index order."
    )]
    async fn lcm_expand(
        &self,
        Parameters(input): Parameters<LcmNodeInput>,
    ) -> Result<CallToolResult, ErrorData> {
        check_lcm_node_input(&input).map_err(validation_err)?;
        let db = self.db.clone();
        let node_id = input.node_id.clone();
        let res = tokio::task::spawn_blocking(move || lcm::expand(&input.node_id, &db))
            .await
            .map_err(|e| internal_err("task_join_failed", format!("join: {e}")))?;
        match res {
            Ok(rows) => Ok(CallToolResult::structured(json!({"turns": rows}))),
            Err(e) if e.is::<LcmNotFound>() => Err(lcm_not_found(&node_id)),
            Err(e) => Err(internal_err(
                "lcm_expand_failed",
                format!("lcm_expand failed: {e}"),
            )),
        }
    }
}

#[tool_handler]
impl ServerHandler for PolymorphServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2025_06_18)
            .with_server_info(Implementation::new(
                "polymorph-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

/// Read a local log file for `compress_log`, bounded to [`MAX_LOG_FILE_BYTES`] and
/// required to be valid UTF-8. The server is local/single-user, so reading a
/// caller-named local path is in-scope (the same trust model as the SQLite cache).
fn read_log_file(path: &str) -> anyhow::Result<String> {
    use std::io::Read;
    let pb = crate::expand_home_path(path);
    let meta = std::fs::metadata(&pb).map_err(|e| anyhow::anyhow!("stat {}: {e}", pb.display()))?;
    if !meta.is_file() {
        anyhow::bail!("{} is not a regular file", pb.display());
    }
    if meta.len() > MAX_LOG_FILE_BYTES {
        anyhow::bail!(
            "{} is {} bytes, exceeds the {} byte cap",
            pb.display(),
            meta.len(),
            MAX_LOG_FILE_BYTES
        );
    }
    let f = std::fs::File::open(&pb).map_err(|e| anyhow::anyhow!("open {}: {e}", pb.display()))?;
    let mut buf = Vec::with_capacity(meta.len() as usize);
    f.take(MAX_LOG_FILE_BYTES)
        .read_to_end(&mut buf)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", pb.display()))?;
    String::from_utf8(buf).map_err(|e| anyhow::anyhow!("{} is not valid UTF-8: {e}", pb.display()))
}

fn validation_err(e: anyhow::Error) -> ErrorData {
    ErrorData::invalid_params(
        format!("invalid arguments: {e}"),
        Some(json!({
            "error": "validation_failed",
            "details": e.to_string(),
            "hint": "fix the tool arguments to match the schema and semantic bounds",
        })),
    )
}

fn lcm_not_found(node_id: &str) -> ErrorData {
    ErrorData::invalid_params(
        "lcm_not_found",
        Some(json!({
            "error": "lcm_not_found",
            "node_id": node_id,
            "hint": "no summary node with that id",
        })),
    )
}

fn internal_err(error: &'static str, message: String) -> ErrorData {
    ErrorData::internal_error(
        message.clone(),
        Some(json!({
            "error": error,
            "details": message,
            "hint": "retry later or inspect server logs",
        })),
    )
}
