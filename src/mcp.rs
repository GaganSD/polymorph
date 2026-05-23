use anyhow::Result;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

use crate::io_guard::{validate_lock_mask_input, BoundedStdin};
use crate::{lock_payload, Language};

/// Minimal JSON-RPC 2.0 over newline-delimited stdio. Implements the three MCP
/// methods needed for a single-tool server.
pub fn serve(grammars_dir: PathBuf) -> Result<()> {
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

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned().unwrap_or(json!({}));

        let response = match method.as_str() {
            "initialize" => handle_initialize(id),
            "tools/list" => handle_tools_list(id),
            "tools/call" => handle_tools_call(id, &params, &grammars_dir),
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
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "polymorph-mcp", "version": env!("CARGO_PKG_VERSION")}
        }
    })
}

fn handle_tools_list(id: Value) -> Value {
    let schema = schemars::schema_for!(crate::io_guard::LockMaskInput);
    let schema_json = serde_json::to_value(&schema).unwrap_or(json!({}));
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [{
                "name": "lock_mask",
                "description": "Produce a per-token boolean mask where true = locked (syntax/keyword) and false = unlocked (eligible for pruning).",
                "inputSchema": schema_json,
            }]
        }
    })
}

fn handle_tools_call(id: Value, params: &Value, grammars_dir: &std::path::Path) -> Value {
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    if tool_name != "lock_mask" {
        return error_response(id, -32602, &format!("unknown tool: {tool_name}"));
    }
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    let input = match validate_lock_mask_input(&arguments) {
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
        Ok(res) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&json!({
                        "tokens": res.token_ids.len(),
                        "mask": res.mask,
                    })).unwrap_or_default(),
                }],
                "structuredContent": {
                    "tokens": res.token_ids.len(),
                    "mask": res.mask,
                }
            }
        }),
        Err(e) => error_response(id, -32000, &format!("lock_payload failed: {e}")),
    }
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}
