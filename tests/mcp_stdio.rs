use std::io::Write;
use std::process::{Command, Stdio};

fn binary_path() -> std::path::PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for integration tests.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_polymorph-mcp"))
}

fn send_messages(messages: &[&str]) -> String {
    let mut child = Command::new(binary_path())
        .env(
            "POLYMORPH_GRAMMARS_DIR",
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars"),
        )
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
