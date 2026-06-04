//! Stdio smoke tests: spawn the actual binary, drive it via the rmcp SDK
//! client over its real stdin/stdout. Verifies the wire protocol end-to-end.
//! The bulk of tool-call coverage lives in `mcp_inproc.rs` for speed; this
//! file only covers properties unique to the binary: real process boot, real
//! stdio transport, and the BoundedAsyncRead payload cap.

use polymorph::io_guard::MAX_PAYLOAD_BYTES;
use rmcp::model::CallToolRequestParams;
use rmcp::ServiceExt;
use serde_json::json;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

fn binary_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_polymorph-mcp"))
}

fn manifest_grammars() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
}

async fn spawn_server() -> (Child, rmcp::service::RunningService<rmcp::RoleClient, ()>) {
    let db_path =
        std::env::temp_dir().join(format!("polymorph-stdio-smoke-{}.db", uuid::Uuid::new_v4()));
    let mut child = Command::new(binary_path())
        .env("POLYMORPH_GRAMMARS_DIR", manifest_grammars())
        .env("POLYMORPH_DB_PATH", &db_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn polymorph-mcp");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let client = ().serve((stdout, stdin)).await.expect("client initialize");
    (child, client)
}

#[tokio::test]
async fn stdio_initialize_handshake() {
    let (mut child, client) = spawn_server().await;
    let info = client.peer_info().expect("peer info after init");
    assert_eq!(info.server_info.name, "polymorph-mcp");
    assert_eq!(
        info.protocol_version,
        rmcp::model::ProtocolVersion::V_2025_06_18
    );
    let _ = client.cancel().await;
    let _ = child.kill().await;
}

#[tokio::test]
async fn stdio_tools_list_advertises_all_six_tools() {
    let (mut child, client) = spawn_server().await;
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
        assert!(
            names.contains(&expected),
            "missing tool {expected}: {names:?}"
        );
    }
    let _ = client.cancel().await;
    let _ = child.kill().await;
}

#[tokio::test]
async fn stdio_lock_mask_round_trip() {
    let (mut child, client) = spawn_server().await;
    let result = client
        .peer()
        .call_tool(
            CallToolRequestParams::new("lock_mask").with_arguments(
                json!({"text": "{\"k\":\"v\"}", "language": "json", "keywords": []})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )
        .await
        .expect("call_tool");
    let sc = result.structured_content.expect("structuredContent");
    // {"k":"v"} → 5 tokens, mask is [true, false, true, false, true]
    assert_eq!(sc["mask"], json!([true, false, true, false, true]));
    let _ = client.cancel().await;
    let _ = child.kill().await;
}

#[tokio::test]
async fn stdio_rejects_oversize_message() {
    // BoundedAsyncRead must trip the cap on a single JSON-RPC line before rmcp
    // allocates/deserializes the enormous arguments object.
    let db_path =
        std::env::temp_dir().join(format!("polymorph-oversize-{}.db", uuid::Uuid::new_v4()));
    let mut child = Command::new(binary_path())
        .env("POLYMORPH_GRAMMARS_DIR", manifest_grammars())
        .env("POLYMORPH_DB_PATH", &db_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let prefix = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lock_mask","arguments":{"text":""#;
    let suffix = br#"","language":"json","keywords":[]}}}"#;
    let chunk = vec![b'a'; 64 * 1024];
    let target_body_bytes = MAX_PAYLOAD_BYTES as usize + 8 * 1024;

    let write_result = async {
        stdin.write_all(prefix).await?;
        let mut written = 0usize;
        while written < target_body_bytes {
            let remaining = target_body_bytes - written;
            let n = remaining.min(chunk.len());
            stdin.write_all(&chunk[..n]).await?;
            written += n;
        }
        stdin.write_all(suffix).await?;
        stdin.write_all(b"\n").await
    };
    let _ = write_result.await;
    drop(stdin);
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("server did not terminate after oversized payload")
        .expect("wait");
    let mut stderr_text = String::new();
    let _ = stderr.read_to_string(&mut stderr_text).await;
    assert!(
        !status.success(),
        "server should exit non-zero when bounded reader trips the cap; stderr={stderr_text}"
    );
    assert!(
        stderr_text.contains("exceeds")
            || stderr_text.contains("Transport")
            || stderr_text.contains("connection closed"),
        "expected bounded-reader/transport failure on stderr, got: {stderr_text}"
    );
}
