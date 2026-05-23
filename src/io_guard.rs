use anyhow::{anyhow, Result};
use jsonschema::JSONSchema;
use once_cell::sync::OnceCell;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};

/// Hard cap on a single MCP message size. JSON-RPC messages above this are
/// rejected at the reader before any parsing — defeats zip-bomb-style inflation
/// since we never allocate beyond this bound.
pub const MAX_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;

/// Wraps stdin (or any reader) with `Read::take` and a line-buffered reader so
/// we can read newline-delimited JSON messages while never reading more than
/// `MAX_PAYLOAD_BYTES * N_messages` total. Per-line cap enforced by
/// `read_message`.
pub struct BoundedStdin<R: Read> {
    inner: BufReader<R>,
}

impl<R: Read> BoundedStdin<R> {
    pub fn new(reader: R) -> Self {
        Self {
            inner: BufReader::new(reader),
        }
    }

    /// Reads one newline-delimited JSON-RPC message, refusing any message that
    /// exceeds `MAX_PAYLOAD_BYTES`. Returns `Ok(None)` on EOF.
    pub fn read_message(&mut self) -> Result<Option<String>> {
        let mut buf = String::new();
        // BufRead::take returns a bounded reader; we read until newline OR cap.
        let mut limited = (&mut self.inner).take(MAX_PAYLOAD_BYTES + 1);
        let mut bytes = Vec::new();
        let _ = limited
            .by_ref()
            .read_until(b'\n', &mut bytes)
            .map_err(|e| anyhow!("stdin read error: {e}"))?;
        if bytes.is_empty() {
            return Ok(None);
        }
        if bytes.len() as u64 > MAX_PAYLOAD_BYTES {
            return Err(anyhow!(
                "incoming message exceeds {MAX_PAYLOAD_BYTES} bytes — refusing"
            ));
        }
        buf.push_str(
            std::str::from_utf8(&bytes).map_err(|e| anyhow!("non-utf8 message: {e}"))?,
        );
        Ok(Some(buf))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LockMaskInput {
    /// The text payload to analyze. Capped by MAX_PAYLOAD_BYTES upstream.
    pub text: String,
    /// "json" or "python".
    pub language: String,
    /// Keyword strings to hard-lock via DAAC.
    #[serde(default)]
    pub keywords: Vec<String>,
}

static LOCK_MASK_SCHEMA: OnceCell<JSONSchema> = OnceCell::new();

fn lock_mask_schema() -> Result<&'static JSONSchema> {
    LOCK_MASK_SCHEMA.get_or_try_init(|| {
        let schema = schemars::schema_for!(LockMaskInput);
        let json = serde_json::to_value(&schema)
            .map_err(|e| anyhow!("schema serialization failed: {e}"))?;
        JSONSchema::options()
            .compile(&json)
            .map_err(|e| anyhow!("schema compile failed: {e}"))
    })
}

/// Validates an incoming JSON value against the LockMaskInput schema BEFORE
/// deserialization. This is the zero-trust boundary: we never construct a Rust
/// struct from unvalidated input.
pub fn validate_lock_mask_input(value: &Value) -> Result<LockMaskInput> {
    let schema = lock_mask_schema()?;
    if let Err(errors) = schema.validate(value) {
        let messages: Vec<String> = errors.map(|e| format!("{e}")).collect();
        return Err(anyhow!("schema validation failed: {}", messages.join("; ")));
    }
    serde_json::from_value(value.clone()).map_err(|e| anyhow!("deserialize failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_valid_input() {
        let v = json!({
            "text": "hello",
            "language": "json",
            "keywords": ["k1", "k2"]
        });
        let parsed = validate_lock_mask_input(&v).unwrap();
        assert_eq!(parsed.language, "json");
        assert_eq!(parsed.keywords.len(), 2);
    }

    #[test]
    fn accepts_input_without_keywords() {
        let v = json!({"text": "hello", "language": "python"});
        let parsed = validate_lock_mask_input(&v).unwrap();
        assert!(parsed.keywords.is_empty());
    }

    #[test]
    fn rejects_missing_text() {
        let v = json!({"language": "json"});
        assert!(validate_lock_mask_input(&v).is_err());
    }

    #[test]
    fn rejects_wrong_type() {
        let v = json!({"text": 123, "language": "json"});
        assert!(validate_lock_mask_input(&v).is_err());
    }

    #[test]
    fn bounded_reader_reads_one_line() {
        let input = b"hello\nworld\n";
        let mut r = BoundedStdin::new(&input[..]);
        let msg = r.read_message().unwrap().unwrap();
        assert_eq!(msg, "hello\n");
        let msg2 = r.read_message().unwrap().unwrap();
        assert_eq!(msg2, "world\n");
        assert!(r.read_message().unwrap().is_none());
    }

    #[test]
    fn bounded_reader_rejects_oversize_message() {
        // Build a line that overflows MAX_PAYLOAD_BYTES.
        let huge = vec![b'a'; (MAX_PAYLOAD_BYTES as usize) + 100];
        let mut r = BoundedStdin::new(&huge[..]);
        let err = r.read_message().expect_err("must reject");
        let msg = format!("{err}");
        assert!(msg.contains("exceeds"), "got: {msg}");
    }

    #[test]
    fn bounded_reader_handles_non_utf8() {
        let input = [b'h', 0xFF, b'\n'];
        let mut r = BoundedStdin::new(&input[..]);
        assert!(r.read_message().is_err());
    }

    #[test]
    fn lock_mask_input_schema_is_object() {
        let schema = schemars::schema_for!(LockMaskInput);
        let v = serde_json::to_value(&schema).unwrap();
        assert_eq!(v["type"], "object");
        assert!(v["required"].as_array().unwrap().iter().any(|r| r == "text"));
        assert!(v["required"].as_array().unwrap().iter().any(|r| r == "language"));
    }
}
