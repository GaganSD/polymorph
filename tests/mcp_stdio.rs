use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

fn binary_path() -> std::path::PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for integration tests.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_polymorph-mcp"))
}

/// Interactive session: spawn the binary, write one request, read one response line.
/// Used when later requests depend on earlier responses (e.g. cache_id round-trip).
struct InteractiveSession {
    child: Child,
}

impl InteractiveSession {
    fn new() -> Self {
        let db_path = std::env::temp_dir().join(format!(
            "polymorph-interactive-{}.db",
            uuid::Uuid::new_v4()
        ));
        let child = Command::new(binary_path())
            .env(
                "POLYMORPH_GRAMMARS_DIR",
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars"),
            )
            .env("POLYMORPH_DB_PATH", &db_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        Self { child }
    }

    fn call(&mut self, request: &str) -> serde_json::Value {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        let stdout = self.child.stdout.as_mut().unwrap();
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).expect("parse response")
    }
}

impl Drop for InteractiveSession {
    fn drop(&mut self) {
        // Closing stdin will cause the server to exit.
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

fn send_messages(messages: &[&str]) -> String {
    // Per-test DB so concurrent cargo test runs don't collide.
    let db_path = std::env::temp_dir().join(format!(
        "polymorph-stdio-test-{}.db",
        uuid::Uuid::new_v4()
    ));
    let mut child = Command::new(binary_path())
        .env(
            "POLYMORPH_GRAMMARS_DIR",
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars"),
        )
        .env("POLYMORPH_DB_PATH", &db_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn polymorph-mcp");

    {
        let stdin = child.stdin.as_mut().unwrap();
        for m in messages {
            writeln!(stdin, "{m}").unwrap();
        }
        // Closing stdin signals EOF.
    }

    let output = child.wait_with_output().expect("wait");
    String::from_utf8(output.stdout).expect("utf8")
}

#[test]
fn mcp_initialize() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#]);
    assert!(out.contains("\"protocolVersion\""));
    assert!(out.contains("\"polymorph-mcp\""));
}

#[test]
fn mcp_tools_list() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#]);
    assert!(out.contains("\"lock_mask\""));
    assert!(out.contains("\"inputSchema\""));
}

#[test]
fn mcp_lock_mask_call() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lock_mask","arguments":{"text":"{\"k\":\"v\"}","language":"json","keywords":[]}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"structuredContent\""));
    assert!(out.contains("\"mask\""));
    // The 5-token JSON `{"k":"v"}` has known shape: braces + colon locked, k/v unlocked.
    assert!(out.contains("[true,false,true,false,true]"));
}

#[test]
fn mcp_lock_mask_python() {
    let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lock_mask","arguments":{"text":"def f():\n    pass\n","language":"python","keywords":[]}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"structuredContent\""));
}

#[test]
fn mcp_invalid_json_returns_parse_error() {
    let out = send_messages(&["not json at all"]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("-32700"));
}

#[test]
fn mcp_missing_required_field_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lock_mask","arguments":{"language":"json"}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("invalid arguments"));
}

#[test]
fn mcp_unsupported_language_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lock_mask","arguments":{"text":"x","language":"rust"}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("unsupported language"));
}

#[test]
fn mcp_unknown_tool_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"not_a_tool","arguments":{}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("unknown tool"));
}

#[test]
fn mcp_unknown_method_returns_method_not_found() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":1,"method":"does/not/exist"}"#]);
    assert!(out.contains("-32601"));
    assert!(out.contains("method not found"));
}

#[test]
fn mcp_ping() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":42,"method":"ping"}"#]);
    assert!(out.contains("\"id\":42"));
    assert!(out.contains("\"result\""));
}

#[test]
fn mcp_notification_produces_no_response() {
    // No `id` field → JSON-RPC notification. Server must not write a response.
    // We send a notification then a real request; only the request gets a reply.
    let out = send_messages(&[
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#,
    ]);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 1, "exactly one response expected, got: {out}");
    assert!(lines[0].contains("\"id\":7"));
}

#[test]
fn mcp_protocol_version_is_current() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#]);
    assert!(out.contains("\"2025-06-18\""), "expected 2025-06-18 protocol version, got: {out}");
}

#[test]
fn mcp_tools_list_advertises_all_six_tools() {
    let out = send_messages(&[r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#]);
    for tool in [
        "lock_mask",
        "compress_array",
        "polymorph_retrieve_cache",
        "lcm_append",
        "lcm_describe",
        "lcm_expand",
    ] {
        assert!(out.contains(tool), "tools/list missing {tool}: {out}");
    }
}

#[test]
fn mcp_compress_array_short_passes_through() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"compress_array","arguments":{"value":[1,2,3]}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"cache_id\":null"));
    assert!(out.contains("\"omitted_count\":0"));
}

#[test]
fn mcp_compress_array_long_then_retrieve_round_trip() {
    let mut arr = String::from("[");
    for i in 0..50 {
        if i > 0 {
            arr.push(',');
        }
        arr.push_str(&format!("{{\"i\":{}}}", i));
    }
    arr.push(']');

    let compress_req = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"compress_array","arguments":{{"value":{}}}}}}}"#,
        arr
    );

    let mut session = InteractiveSession::new();
    let resp1 = session.call(&compress_req);
    let cache_id = resp1["result"]["structuredContent"]["cache_id"]
        .as_str()
        .expect("cache_id from compress")
        .to_string();
    assert_eq!(
        resp1["result"]["structuredContent"]["omitted_count"], 44
    );

    let retrieve_req = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"polymorph_retrieve_cache","arguments":{{"cache_id":"{}"}}}}}}"#,
        cache_id
    );
    let resp2 = session.call(&retrieve_req);
    let recovered = resp2["result"]["structuredContent"]["value"]
        .as_array()
        .expect("array recovered");
    assert_eq!(recovered.len(), 44);
}

#[test]
fn mcp_retrieve_unknown_cache_returns_structured_error() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"polymorph_retrieve_cache","arguments":{"cache_id":"does-not-exist"}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("cache_miss"), "got: {out}");
    assert!(out.contains("does-not-exist"));
}

#[test]
fn mcp_lcm_append_describe_expand_round_trip() {
    let big = "lorem ipsum dolor sit amet ".repeat(50);
    let mk_append = |id: u32| {
        format!(
            r#"{{"jsonrpc":"2.0","id":{},"method":"tools/call","params":{{"name":"lcm_append","arguments":{{"conversation_id":"c1","role":"user","content":"{}","soft_threshold":50}}}}}}"#,
            id, big
        )
    };

    let mut session = InteractiveSession::new();
    let _resp_a = session.call(&mk_append(1));
    let resp_b = session.call(&mk_append(2));
    let node_id = resp_b["result"]["structuredContent"]["archived_node_id"]
        .as_str()
        .expect("archive must fire on the second append (synchronous)")
        .to_string();

    let req_desc = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"lcm_describe","arguments":{{"node_id":"{}"}}}}}}"#,
        node_id
    );
    let desc = session.call(&req_desc);
    assert_eq!(desc["result"]["structuredContent"]["depth"], 0);
    assert!(
        desc["result"]["structuredContent"]["child_count"]
            .as_u64()
            .unwrap()
            >= 1
    );

    let req_exp = format!(
        r#"{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"lcm_expand","arguments":{{"node_id":"{}"}}}}}}"#,
        node_id
    );
    let exp = session.call(&req_exp);
    let turns = exp["result"]["structuredContent"]["turns"]
        .as_array()
        .unwrap();
    assert!(!turns.is_empty());
    assert!(turns[0]["content"].as_str().unwrap().contains("lorem"));
}

#[test]
fn mcp_lcm_describe_unknown_returns_not_found_data() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lcm_describe","arguments":{"node_id":"nope"}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("lcm_not_found"));
    assert!(out.contains("nope"));
}

#[test]
fn mcp_compress_array_missing_value_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"compress_array","arguments":{"head_keep":3}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("invalid arguments"));
}

#[test]
fn mcp_retrieve_cache_missing_id_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"polymorph_retrieve_cache","arguments":{}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("invalid arguments"));
}

#[test]
fn mcp_lcm_append_missing_content_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lcm_append","arguments":{"conversation_id":"c1","role":"user"}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("invalid arguments"));
}

#[test]
fn mcp_lcm_describe_missing_node_id_rejected() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lcm_describe","arguments":{}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"error\""));
    assert!(out.contains("invalid arguments"));
}

#[test]
fn mcp_lock_mask_response_includes_drop_mask() {
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lock_mask","arguments":{"text":"{\"k\":\"v\"}","language":"json","keywords":[]}}}"#;
    let out = send_messages(&[req]);
    assert!(out.contains("\"drop_mask\""));
    assert!(out.contains("\"kept_tokens\""));
}

#[test]
fn mcp_multiple_messages_in_sequence() {
    let out = send_messages(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"ping"}"#,
    ]);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"id\":1"));
    assert!(lines[1].contains("\"id\":2"));
    assert!(lines[2].contains("\"id\":3"));
}
