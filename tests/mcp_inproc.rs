//! In-process MCP tests. Construct `PolymorphServer` directly, connect via an
//! in-memory `tokio::io::duplex` pair, and drive it with the rmcp SDK client.
//! No subprocess, no stdin/stdout, no temp file races on /tmp races — fast.

use polymorph::db;
use polymorph::io_guard::{
    MAX_CONTENT_LEN, MAX_ID_LEN, MAX_JSON_ARRAY_ITEMS, MAX_KEYWORDS, MAX_KEYWORD_ITEM_LEN,
    MAX_TEXT_LEN,
};
use polymorph::mcp::PolymorphServer;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RoleServer, RunningService};
use rmcp::ServiceExt;
use serde_json::json;
use serde_json::Value;
use std::time::Duration;
use tempfile::TempDir;

fn manifest_grammars() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
}

async fn harness() -> (
    RunningService<RoleClient, ()>,
    RunningService<RoleServer, PolymorphServer>,
    TempDir,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = db::open_pool(&dir.path().join("polymorph.db")).expect("open_pool");
    let server = PolymorphServer::new(db, manifest_grammars());
    let (s_io, c_io) = tokio::io::duplex(64 * 1024);
    let server_fut = server.serve(s_io);
    let client_fut = ().serve(c_io);
    let (server, client) = tokio::join!(server_fut, client_fut);
    (
        client.expect("client init"),
        server.expect("server init"),
        dir,
    )
}

fn req(name: &str, v: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(name.to_string()).with_arguments(v.as_object().unwrap().clone())
}

fn tool_schema(tools: &rmcp::model::ListToolsResult, name: &str) -> Value {
    let tool = tools
        .tools
        .iter()
        .find(|t| t.name.as_ref() == name)
        .unwrap_or_else(|| panic!("missing tool {name}"));
    serde_json::to_value(&*tool.input_schema).expect("schema serialize")
}

fn nested_element(i: usize, depth: usize) -> Value {
    let mut value = json!({"i": i});
    for level in 0..depth {
        value = json!({"level": level, "child": value});
    }
    value
}

fn filler(words: usize) -> String {
    (0..words)
        .map(|i| format!("token{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn call(
    client: &RunningService<RoleClient, ()>,
    name: &str,
    v: Value,
) -> rmcp::model::CallToolResult {
    client
        .peer()
        .call_tool(req(name, v))
        .await
        .expect("call_tool ok")
}

async fn call_err(
    client: &RunningService<RoleClient, ()>,
    name: &str,
    v: Value,
) -> rmcp::ErrorData {
    let e = client
        .peer()
        .call_tool(req(name, v))
        .await
        .expect_err("call_tool err");
    match e {
        rmcp::service::ServiceError::McpError(d) => d,
        other => panic!("expected McpError, got: {other:?}"),
    }
}

#[tokio::test]
async fn lock_mask_locks_structural_tokens() {
    let (client, _server, _dir) = harness().await;
    let r = call(
        &client,
        "lock_mask",
        json!({"text": "{\"k\":\"v\"}", "language": "json", "keywords": []}),
    )
    .await;
    let sc = r.structured_content.expect("structuredContent");
    assert_eq!(sc["mask"], json!([true, false, true, false, true]));
    assert!(sc["kept_tokens"].as_u64().unwrap() > 0);
    assert!(sc["drop_mask"].as_array().is_some());
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lock_mask_python_works() {
    let (client, _server, _dir) = harness().await;
    let r = call(
        &client,
        "lock_mask",
        json!({"text": "def f():\n    pass\n", "language": "python", "keywords": []}),
    )
    .await;
    assert!(r.structured_content.is_some());
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lock_mask_missing_text_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "lock_mask", json!({"language": "json"})).await;
    let msg = format!("{}", err.message);
    assert!(
        msg.to_lowercase().contains("missing") || msg.contains("text"),
        "got: {msg}"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lock_mask_unsupported_language_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "lock_mask",
        json!({"text": "x", "language": "rust"}),
    )
    .await;
    assert!(format!("{}", err.message).contains("unsupported language"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn unknown_tool_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "not_a_tool", json!({})).await;
    let msg = format!("{}", err.message);
    assert!(msg.to_lowercase().contains("tool"), "got: {msg}");
    let _ = client.cancel().await;
}

#[tokio::test]
async fn compress_array_short_passes_through() {
    let (client, _server, _dir) = harness().await;
    let r = call(&client, "compress_array", json!({"value": [1, 2, 3]})).await;
    let sc = r.structured_content.unwrap();
    assert_eq!(sc["cache_id"], Value::Null);
    assert_eq!(sc["omitted_count"], 0);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn compress_then_retrieve_round_trip() {
    let (client, _server, _dir) = harness().await;
    let mut arr = Vec::new();
    for i in 0..50 {
        arr.push(json!({"i": i}));
    }
    let resp = call(&client, "compress_array", json!({"value": arr})).await;
    let sc = resp.structured_content.unwrap();
    let cache_id = sc["cache_id"].as_str().expect("cache_id").to_string();
    assert_eq!(sc["omitted_count"], 44);

    let resp2 = call(
        &client,
        "polymorph_retrieve_cache",
        json!({"cache_id": cache_id}),
    )
    .await;
    let recovered = resp2.structured_content.unwrap();
    assert_eq!(recovered["value"].as_array().unwrap().len(), 44);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn compress_array_honors_explicit_edges() {
    let (client, _server, _dir) = harness().await;
    let arr: Vec<Value> = (0..30).map(|i| json!({"i": i})).collect();
    let resp = call(
        &client,
        "compress_array",
        json!({"value": arr, "head_keep": 2, "tail_keep": 1, "cache": false}),
    )
    .await;
    let sc = resp.structured_content.unwrap();
    let compressed = sc["compressed"].as_array().unwrap();
    assert_eq!(sc["omitted_count"], 27);
    assert_eq!(compressed.len(), 4, "head + summary + tail");
    assert_eq!(compressed[0]["i"], 0);
    assert_eq!(compressed[1]["i"], 1);
    assert_eq!(compressed[3]["i"], 29);
    assert_eq!(compressed[2]["__omitted_count"], 27);
    assert_eq!(compressed[2]["__polymorph_cache_id"], Value::Null);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn ccr_slicing_preserves_deep_json_edges_and_valid_commas() {
    let (client, _server, _dir) = harness().await;
    let original: Vec<Value> = (0..80).map(|i| nested_element(i, 48)).collect();

    let resp = call(
        &client,
        "compress_array",
        json!({
            "value": original.clone(),
            "head_keep": 4,
            "tail_keep": 4,
            "cache": false
        }),
    )
    .await;
    let sc = resp.structured_content.unwrap();
    assert_eq!(sc["omitted_count"], 72);

    let compressed = sc["compressed"].as_array().unwrap();
    assert_eq!(compressed.len(), 9, "head + summary + tail");
    for i in 0..4 {
        assert_eq!(compressed[i], original[i], "head element {i} changed");
    }
    for i in 0..4 {
        assert_eq!(
            compressed[5 + i],
            original[original.len() - 4 + i],
            "tail element {i} changed"
        );
    }
    assert_eq!(compressed[4]["__omitted_count"], 72);

    let encoded = serde_json::to_string(&sc["compressed"]).expect("serialize compressed JSON");
    assert!(
        encoded.matches(',').count() >= 8,
        "compressed array lost structural separators: {encoded}"
    );
    let reparsed: Value = serde_json::from_str(&encoded).expect("valid JSON after slicing");
    assert_eq!(reparsed, sc["compressed"]);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn retrieve_unknown_cache_returns_structured_error() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "polymorph_retrieve_cache",
        json!({"cache_id": "does-not-exist"}),
    )
    .await;
    assert!(format!("{}", err.message).contains("cache_miss"));
    let data = err.data.as_ref().expect("error data");
    assert_eq!(data["cache_id"], "does-not-exist");
    assert!(data["hint"].as_str().unwrap().contains("cache"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn token_mask_collision_locks_daac_ast_and_lamr_overlap() {
    let (client, _server, _dir) = harness().await;
    let candidate = (0..10_000)
        .map(|n| n.to_string())
        .find(|text| {
            let (ids, spans) = polymorph::tokenizer::token_spans(text).unwrap();
            ids.len() == 1
                && spans.as_slice() == &[(0, text.len())]
                && polymorph::lamr::dummy_lamr_forward_pass(&ids)[0]
        })
        .expect("single-token JSON number that mock LaMR would drop");

    let r = call(
        &client,
        "lock_mask",
        json!({
            "text": candidate.clone(),
            "language": "json",
            "keywords": [candidate]
        }),
    )
    .await;
    let sc = r.structured_content.unwrap();
    assert_eq!(sc["tokens"], 1);
    assert_eq!(sc["mask"], json!([true]), "AST and DAAC must lock the token");
    assert_eq!(
        sc["drop_mask"],
        json!([false]),
        "LaMR-targeted token must not be dropped once locked"
    );
    assert_eq!(sc["kept_tokens"], 1);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn compress_array_missing_value_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "compress_array", json!({"head_keep": 3})).await;
    let msg = format!("{}", err.message);
    assert!(msg.to_lowercase().contains("value") || msg.to_lowercase().contains("missing"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn retrieve_cache_missing_id_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "polymorph_retrieve_cache", json!({})).await;
    let msg = format!("{}", err.message);
    assert!(msg.to_lowercase().contains("cache_id") || msg.to_lowercase().contains("missing"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn retrieve_cache_oversized_id_rejected_by_semantic_validator() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "polymorph_retrieve_cache",
        json!({"cache_id": "x".repeat(MAX_ID_LEN + 1)}),
    )
    .await;
    assert_eq!(err.code.0, -32602);
    let msg = format!("{}", err.message);
    assert!(msg.contains("cache_id exceeds max length"), "got: {msg}");
    assert!(
        err.data.is_none(),
        "validation errors should not include data"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn compress_array_rejects_pathological_edges_over_mcp() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "compress_array",
        json!({"value": [1, 2, 3], "head_keep": MAX_JSON_ARRAY_ITEMS, "tail_keep": 1}),
    )
    .await;
    assert_eq!(err.code.0, -32602);
    let msg = format!("{}", err.message);
    assert!(
        msg.contains("head_keep + tail_keep exceeds max items"),
        "got: {msg}"
    );
    assert!(
        err.data.is_none(),
        "validation errors should not include data"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_append_describe_expand_round_trip() {
    let (client, _server, _dir) = harness().await;
    let big = "lorem ipsum dolor sit amet ".repeat(50);
    let args1 = json!({
        "conversation_id": "c1",
        "role": "user",
        "content": big.clone(),
        "soft_threshold": 50
    });
    let _ = call(&client, "lcm_append", args1.clone()).await;
    let r2 = call(&client, "lcm_append", args1).await;
    let node_id = r2.structured_content.unwrap()["archived_node_id"]
        .as_str()
        .expect("archived_node_id on second append")
        .to_string();

    let desc = call(&client, "lcm_describe", json!({"node_id": node_id.clone()})).await;
    let dsc = desc.structured_content.unwrap();
    assert_eq!(dsc["depth"], 0);
    assert!(dsc["child_count"].as_u64().unwrap() >= 1);

    let exp = call(&client, "lcm_expand", json!({"node_id": node_id})).await;
    let turns = exp.structured_content.unwrap()["turns"]
        .as_array()
        .unwrap()
        .clone();
    assert!(!turns.is_empty());
    assert!(turns[0]["content"].as_str().unwrap().contains("lorem"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_concurrency_stress_archives_without_deadlocking_db_worker() {
    let (client, _server, _dir) = harness().await;
    let peer = client.peer().clone();
    let content = filler(90);
    let conversation_id = format!("stress-{}", uuid::Uuid::new_v4());

    let mut tasks = Vec::new();
    for i in 0..64 {
        let peer = peer.clone();
        let content = format!("{content} turn-{i}");
        let conversation_id = conversation_id.clone();
        tasks.push(tokio::spawn(async move {
            peer.call_tool(req(
                "lcm_append",
                json!({
                    "conversation_id": conversation_id,
                    "role": if i % 2 == 0 { "user" } else { "assistant" },
                    "content": content,
                    "soft_threshold": 80
                }),
            ))
            .await
        }));
    }

    let archived = tokio::time::timeout(Duration::from_secs(10), async {
        let mut archived = Vec::new();
        for task in tasks {
            let result = task.await.expect("task join").expect("lcm_append");
            let sc = result.structured_content.expect("structuredContent");
            if let Some(node_id) = sc["archived_node_id"].as_str() {
                archived.push(node_id.to_string());
            }
        }
        archived
    })
    .await
    .expect("LCM stress test timed out; possible DB worker deadlock");

    assert!(
        archived.len() >= 16,
        "expected many rapid overflow archives, got {}",
        archived.len()
    );
    let desc = call(
        &client,
        "lcm_describe",
        json!({"node_id": archived.last().unwrap()}),
    )
    .await;
    assert!(desc.structured_content.unwrap()["child_count"]
        .as_u64()
        .unwrap()
        > 0);
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_describe_unknown_returns_not_found_data() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "lcm_describe", json!({"node_id": "nope"})).await;
    assert!(format!("{}", err.message).contains("lcm_not_found"));
    let data = err.data.as_ref().expect("error data");
    assert_eq!(data["node_id"], "nope");
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_expand_unknown_returns_not_found_data() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "lcm_expand", json!({"node_id": "nope"})).await;
    assert!(format!("{}", err.message).contains("lcm_not_found"));
    assert_eq!(err.code.0, -32602);
    let data = err.data.as_ref().expect("error data");
    assert_eq!(data["node_id"], "nope");
    assert_eq!(data["hint"], "no summary node with that id");
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_append_missing_content_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "lcm_append",
        json!({"conversation_id": "c1", "role": "user"}),
    )
    .await;
    let msg = format!("{}", err.message);
    assert!(msg.to_lowercase().contains("content") || msg.to_lowercase().contains("missing"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_append_oversized_content_rejected_by_semantic_validator() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "lcm_append",
        json!({
            "conversation_id": "c1",
            "role": "user",
            "content": "a".repeat(MAX_CONTENT_LEN + 1)
        }),
    )
    .await;
    assert_eq!(err.code.0, -32602);
    let msg = format!("{}", err.message);
    assert!(msg.contains("content exceeds max length"), "got: {msg}");
    assert!(
        err.data.is_none(),
        "validation errors should not include data"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_describe_missing_node_id_rejected() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(&client, "lcm_describe", json!({})).await;
    let msg = format!("{}", err.message);
    assert!(msg.to_lowercase().contains("node_id") || msg.to_lowercase().contains("missing"));
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lcm_describe_oversized_node_id_rejected_by_semantic_validator() {
    let (client, _server, _dir) = harness().await;
    let err = call_err(
        &client,
        "lcm_describe",
        json!({"node_id": "x".repeat(MAX_ID_LEN + 1)}),
    )
    .await;
    assert_eq!(err.code.0, -32602);
    let msg = format!("{}", err.message);
    assert!(msg.contains("node_id exceeds max length"), "got: {msg}");
    assert!(
        err.data.is_none(),
        "validation errors should not include data"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn lock_mask_oversized_keywords_rejected_by_semantic_validator() {
    let (client, _server, _dir) = harness().await;
    let keywords: Vec<String> = (0..MAX_KEYWORDS + 1).map(|i| format!("k{i}")).collect();
    let err = call_err(
        &client,
        "lock_mask",
        json!({"text": "x", "language": "json", "keywords": keywords}),
    )
    .await;
    assert_eq!(err.code.0, -32602);
    let msg = format!("{}", err.message);
    assert!(msg.contains("keywords exceeds max count"), "got: {msg}");
    assert!(
        err.data.is_none(),
        "validation errors should not include data"
    );
    let _ = client.cancel().await;
}

#[tokio::test]
async fn tools_list_advertises_all_six() {
    let (client, _server, _dir) = harness().await;
    let tools = client
        .list_tools(Default::default())
        .await
        .expect("list_tools");
    let names: Vec<&str> = tools.tools.iter().map(|t| t.name.as_ref()).collect();
    for expected in [
        "lock_mask",
        "compress_array",
        "polymorph_retrieve_cache",
        "lcm_append",
        "lcm_describe",
        "lcm_expand",
    ] {
        assert!(names.contains(&expected), "missing {expected}: {names:?}");
    }
    // Each tool publishes an inputSchema object.
    for t in &tools.tools {
        let schema = serde_json::to_value(&*t.input_schema).expect("schema serialize");
        assert_eq!(schema["type"], "object", "tool {} schema: {schema}", t.name);
    }
    let _ = client.cancel().await;
}

#[tokio::test]
async fn tools_list_advertises_schema_bounds() {
    let (client, _server, _dir) = harness().await;
    let tools = client
        .list_tools(Default::default())
        .await
        .expect("list_tools");

    let lock_mask = tool_schema(&tools, "lock_mask");
    assert_eq!(
        lock_mask["properties"]["text"]["maxLength"],
        MAX_TEXT_LEN as u64
    );
    assert_eq!(lock_mask["properties"]["language"]["maxLength"], 16);
    assert_eq!(
        lock_mask["properties"]["keywords"]["maxItems"],
        MAX_KEYWORDS as u64
    );
    assert_eq!(
        lock_mask["properties"]["keywords"]["items"]["maxLength"],
        MAX_KEYWORD_ITEM_LEN as u64
    );

    let retrieve = tool_schema(&tools, "polymorph_retrieve_cache");
    assert_eq!(
        retrieve["properties"]["cache_id"]["maxLength"],
        MAX_ID_LEN as u64
    );

    let append = tool_schema(&tools, "lcm_append");
    assert_eq!(
        append["properties"]["conversation_id"]["maxLength"],
        MAX_ID_LEN as u64
    );
    assert_eq!(append["properties"]["role"]["maxLength"], 64);
    assert_eq!(
        append["properties"]["content"]["maxLength"],
        MAX_CONTENT_LEN as u64
    );

    let describe = tool_schema(&tools, "lcm_describe");
    assert_eq!(
        describe["properties"]["node_id"]["maxLength"],
        MAX_ID_LEN as u64
    );

    let expand = tool_schema(&tools, "lcm_expand");
    assert_eq!(
        expand["properties"]["node_id"]["maxLength"],
        MAX_ID_LEN as u64
    );

    let _ = client.cancel().await;
}

#[tokio::test]
async fn lock_mask_emits_drop_mask_and_kept_tokens() {
    let (client, _server, _dir) = harness().await;
    let r = call(
        &client,
        "lock_mask",
        json!({"text": "{\"k\":\"v\"}", "language": "json", "keywords": []}),
    )
    .await;
    let sc = r.structured_content.unwrap();
    assert!(sc["drop_mask"].is_array());
    assert!(sc["kept_tokens"].is_number());
    let _ = client.cancel().await;
}

#[tokio::test]
async fn structured_content_has_matching_text_payload() {
    let (client, _server, _dir) = harness().await;
    let r = call(&client, "compress_array", json!({"value": [1, 2, 3]})).await;
    let sc = r.structured_content.clone().expect("structuredContent");
    let text = r
        .content
        .first()
        .and_then(|c| c.as_text())
        .expect("text content")
        .text
        .clone();
    assert_eq!(text, sc.to_string());
    let _ = client.cancel().await;
}
