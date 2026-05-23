use anyhow::Result;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

use crate::ccr::{self, CacheMiss, CcrOpts};
use crate::db::DbHandle;
use crate::io_guard::{
    validate_compress_array_input_strict, validate_lcm_append_input_strict,
    validate_lcm_node_input_strict, validate_lock_mask_input_strict,
    validate_retrieve_cache_input_strict, BoundedStdin, CompressArrayInput, LcmAppendInput,
    LcmNodeInput, RetrieveCacheInput, MAX_MASK_TOKENS,
};
use crate::lcm::{self, NotFound as LcmNotFound, DEFAULT_SOFT_THRESHOLD};
use crate::{lock_payload, Language};

/// Server-wide state threaded into each tool handler. Holds the SQLite actor
/// handle, which is cheaply cloneable and owns DB access through one worker.
#[derive(Clone)]
pub struct AppState {
    pub db: DbHandle,
    pub grammars_dir: PathBuf,
}

/// JSON-RPC 2.0 over newline-delimited stdio.
pub fn serve(state: AppState) -> Result<()> {
    let stdin = std::io::stdin().lock();
    let mut reader = BoundedStdin::new(stdin);
    let mut stdout = std::io::stdout().lock();

    while let Some(line) = reader.read_message()? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                write_response(
                    &mut stdout,
                    error_response(Value::Null, -32700, &format!("parse error: {e}")),
                )?;
                continue;
            }
        };

        let is_notification = req.get("id").is_none();
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned().unwrap_or(json!({}));

        if is_notification {
            continue;
        }

        let response = match method.as_str() {
            "initialize" => handle_initialize(id),
            "tools/list" => handle_tools_list(id),
            "tools/call" => handle_tools_call(id, &params, &state),
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
            other => error_response(id, -32601, &format!("method not found: {other}")),
        };

        write_response(&mut stdout, response)?;
    }

    Ok(())
}

fn write_response<W: Write>(w: &mut W, v: Value) -> Result<()> {
    let s = serde_json::to_string(&v)?;
    w.write_all(s.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

fn handle_initialize(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "polymorph-mcp", "version": env!("CARGO_PKG_VERSION")}
        }
    })
}

fn tool_descriptor<S: schemars::JsonSchema>(name: &str, description: &str) -> Value {
    let schema = schemars::schema_for!(S);
    let schema_json = serde_json::to_value(&schema).unwrap_or(json!({}));
    json!({
        "name": name,
        "description": description,
        "inputSchema": schema_json,
    })
}

fn handle_tools_list(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                tool_descriptor::<crate::io_guard::LockMaskInput>(
                    "lock_mask",
                    "Produce a per-token lock+drop mask. lock_mask[i]=true means structural/keyword (must keep); drop_mask[i]=true means the mock LaMR pruner suggests dropping this unlocked token.",
                ),
                tool_descriptor::<CompressArrayInput>(
                    "compress_array",
                    "Compress a large JSON array by keeping head + tail elements and stashing the middle in the local SQLite cache. Returns the compressed array and a cache_id for retrieval.",
                ),
                tool_descriptor::<RetrieveCacheInput>(
                    "polymorph_retrieve_cache",
                    "Retrieve the original middle slice of a previously-compressed JSON array by its cache_id.",
                ),
                tool_descriptor::<LcmAppendInput>(
                    "lcm_append",
                    "Append a conversational turn to the LCM store. If the conversation's active token count exceeds soft_threshold (default 80000), the oldest turns are archived into a Depth-0 summary node.",
                ),
                tool_descriptor::<LcmNodeInput>(
                    "lcm_describe",
                    "Return metadata about an archived summary node (child_count, total_tokens, roles, created_at).",
                ),
                tool_descriptor::<LcmNodeInput>(
                    "lcm_expand",
                    "Return the original verbatim turns archived under a summary node, in turn-index order.",
                ),
            ]
        }
    })
}

fn handle_tools_call(id: Value, params: &Value, state: &AppState) -> Value {
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match tool_name {
        "lock_mask" => call_lock_mask(id, &arguments, &state.grammars_dir),
        "compress_array" => call_compress_array(id, &arguments, &state.db),
        "polymorph_retrieve_cache" => call_retrieve_cache(id, &arguments, &state.db),
        "lcm_append" => call_lcm_append(id, &arguments, state),
        "lcm_describe" => call_lcm_describe(id, &arguments, &state.db),
        "lcm_expand" => call_lcm_expand(id, &arguments, &state.db),
        other => error_response(id, -32602, &format!("unknown tool: {other}")),
    }
}

fn call_lock_mask(id: Value, arguments: &Value, grammars_dir: &std::path::Path) -> Value {
    let input = match validate_lock_mask_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };

    let lang = match Language::from_str(&input.language) {
        Some(l) => l,
        None => {
            return error_response(
                id,
                -32602,
                &format!("unsupported language: {}", input.language),
            )
        }
    };

    match lock_payload(&input.text, lang, &input.keywords, grammars_dir) {
        Ok(res) => {
            if res.mask.len() > MAX_MASK_TOKENS {
                return error_response(
                    id,
                    -32602,
                    &format!(
                        "lock_mask output exceeds max token count ({MAX_MASK_TOKENS}) — reduce input text"
                    ),
                );
            }
            let structured = json!({
                "tokens": res.token_ids.len(),
                "kept_tokens": res.kept_tokens,
                "mask": res.mask,
                "drop_mask": res.drop_mask,
            });
            tool_ok(id, &structured)
        }
        Err(e) => error_response(id, -32000, &format!("lock_payload failed: {e}")),
    }
}

fn call_compress_array(id: Value, arguments: &Value, db: &DbHandle) -> Value {
    let input = match validate_compress_array_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };

    let mut opts = CcrOpts::default();
    if let Some(h) = input.head_keep {
        opts.head_keep = h;
    }
    if let Some(t) = input.tail_keep {
        opts.tail_keep = t;
    }

    match ccr::compress_array(input.value, opts, db, input.cache) {
        Ok(res) => tool_ok(
            id,
            &json!({
                "compressed": res.compressed,
                "cache_id": res.cache_id,
                "omitted_count": res.omitted_count,
            }),
        ),
        Err(e) => error_response(id, -32000, &format!("compress_array failed: {e}")),
    }
}

fn call_retrieve_cache(id: Value, arguments: &Value, db: &DbHandle) -> Value {
    let input = match validate_retrieve_cache_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };

    match ccr::retrieve(&input.cache_id, db) {
        Ok(value) => tool_ok(id, &json!({"value": value})),
        Err(e) if e.is::<CacheMiss>() => error_response_with_data(
            id,
            -32602,
            "cache_miss",
            json!({
                "cache_id": input.cache_id,
                "hint": "cache entry expired or never existed",
            }),
        ),
        Err(e) => error_response(id, -32000, &format!("retrieve failed: {e}")),
    }
}

fn call_lcm_append(id: Value, arguments: &Value, state: &AppState) -> Value {
    let input = match validate_lcm_append_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };
    let threshold = input.soft_threshold.unwrap_or(DEFAULT_SOFT_THRESHOLD);

    let result = match lcm::append_and_maybe_archive(
        &input.conversation_id,
        &input.role,
        &input.content,
        threshold,
        &state.db,
    ) {
        Ok(r) => r,
        Err(e) => return error_response(id, -32000, &format!("lcm_append failed: {e}")),
    };

    tool_ok(
        id,
        &json!({
            "turn_id": result.turn_id,
            "turn_index": result.turn_index,
            "tokens": result.tokens,
            "archived_node_id": result.archived_node_id,
        }),
    )
}

fn call_lcm_describe(id: Value, arguments: &Value, db: &DbHandle) -> Value {
    let input = match validate_lcm_node_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };
    match lcm::describe(&input.node_id, db) {
        Ok(meta) => tool_ok(id, &serde_json::to_value(&meta).unwrap_or(Value::Null)),
        Err(e) if e.is::<LcmNotFound>() => error_response_with_data(
            id,
            -32602,
            "lcm_not_found",
            json!({"node_id": input.node_id, "hint": "no summary node with that id"}),
        ),
        Err(e) => error_response(id, -32000, &format!("lcm_describe failed: {e}")),
    }
}

fn call_lcm_expand(id: Value, arguments: &Value, db: &DbHandle) -> Value {
    let input = match validate_lcm_node_input_strict(arguments) {
        Ok(v) => v,
        Err(e) => return error_response(id, -32602, &format!("invalid arguments: {e}")),
    };
    match lcm::expand(&input.node_id, db) {
        Ok(rows) => tool_ok(id, &json!({"turns": rows})),
        Err(e) if e.is::<LcmNotFound>() => error_response_with_data(
            id,
            -32602,
            "lcm_not_found",
            json!({"node_id": input.node_id, "hint": "no summary node with that id"}),
        ),
        Err(e) => error_response(id, -32000, &format!("lcm_expand failed: {e}")),
    }
}

fn tool_ok(id: Value, structured: &Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{
                "type": "text",
                "text": serde_json::to_string(structured).unwrap_or_default(),
            }],
            "structuredContent": structured,
        }
    })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

fn error_response_with_data(id: Value, code: i64, message: &str, data: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message, "data": data}
    })
}
