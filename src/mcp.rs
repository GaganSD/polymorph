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
use crate::io_guard::{
    check_compress_array_input, check_lcm_append_input, check_lcm_node_input,
    check_lock_mask_input, check_retrieve_cache_input, CompressArrayInput, LcmAppendInput,
    LcmNodeInput, LockMaskInput, RetrieveCacheInput, MAX_MASK_TOKENS,
};
use crate::lcm::{self, NotFound as LcmNotFound, DEFAULT_SOFT_THRESHOLD};
use crate::{lock_payload, Language};

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
            ErrorData::invalid_params(
                format!("unsupported language: {}", input.language),
                None,
            )
        })?;
        let grammars = self.grammars_dir.clone();
        let res = tokio::task::spawn_blocking(move || {
            lock_payload(&input.text, lang, &input.keywords, grammars.as_path())
        })
        .await
        .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?
        .map_err(|e| ErrorData::internal_error(format!("lock_payload failed: {e}"), None))?;
        if res.mask.len() > MAX_MASK_TOKENS {
            return Err(ErrorData::invalid_params(
                format!("lock_mask output exceeds max token count ({MAX_MASK_TOKENS}) \u{2014} reduce input text"),
                None,
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
            .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?
            .map_err(|e| ErrorData::internal_error(format!("compress_array failed: {e}"), None))?;
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
            .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?;
        match res {
            Ok(value) => Ok(CallToolResult::structured(json!({"value": value}))),
            Err(e) if e.is::<CacheMiss>() => Err(ErrorData::invalid_params(
                "cache_miss",
                Some(json!({
                    "cache_id": cache_id,
                    "hint": "cache entry expired or never existed",
                })),
            )),
            Err(e) => Err(ErrorData::internal_error(
                format!("retrieve failed: {e}"),
                None,
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
        .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?
        .map_err(|e| ErrorData::internal_error(format!("lcm_append failed: {e}"), None))?;
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
            .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?;
        match res {
            Ok(meta) => Ok(CallToolResult::structured(
                serde_json::to_value(&meta).unwrap_or(Value::Null),
            )),
            Err(e) if e.is::<LcmNotFound>() => Err(lcm_not_found(&node_id)),
            Err(e) => Err(ErrorData::internal_error(
                format!("lcm_describe failed: {e}"),
                None,
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
            .map_err(|e| ErrorData::internal_error(format!("join: {e}"), None))?;
        match res {
            Ok(rows) => Ok(CallToolResult::structured(json!({"turns": rows}))),
            Err(e) if e.is::<LcmNotFound>() => Err(lcm_not_found(&node_id)),
            Err(e) => Err(ErrorData::internal_error(
                format!("lcm_expand failed: {e}"),
                None,
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

fn validation_err(e: anyhow::Error) -> ErrorData {
    ErrorData::invalid_params(format!("invalid arguments: {e}"), None)
}

fn lcm_not_found(node_id: &str) -> ErrorData {
    ErrorData::invalid_params(
        "lcm_not_found",
        Some(json!({"node_id": node_id, "hint": "no summary node with that id"})),
    )
}
