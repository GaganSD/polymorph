use anyhow::{anyhow, Result};
use jsonschema::JSONSchema;
use once_cell::sync::OnceCell;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};

/// Hard cap on a single MCP message size. JSON-RPC messages above this are
/// rejected at the reader before any parsing — defeats zip-bomb-style inflation
/// since we never allocate beyond this bound.
pub const MAX_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum tokens returned in a lock_mask response (prevents egress amplification).
pub const MAX_MASK_TOKENS: usize = 262_144;

/// Zero-trust field bounds advertised in tool schemas.
pub const MAX_TEXT_LEN: usize = 1_048_576;
pub const MAX_CONTENT_LEN: usize = 1_048_576;
pub const MAX_KEYWORDS: usize = 256;
pub const MAX_KEYWORD_ITEM_LEN: usize = 4_096;
pub const MAX_ID_LEN: usize = 256;
pub const MAX_JSON_ARRAY_ITEMS: usize = 100_000;

/// Wraps stdin (or any reader) with `Read::take` and a line-buffered reader so
/// we can read newline-delimited JSON messages while never reading more than
/// `MAX_PAYLOAD_BYTES` per line.
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
    #[schemars(length(max = 1048576))]
    pub text: String,
    #[schemars(length(max = 16))]
    pub language: String,
    #[schemars(length(max = 256))]
    #[serde(default)]
    pub keywords: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompressArrayInput {
    pub value: serde_json::Value,
    #[serde(default)]
    pub head_keep: Option<usize>,
    #[serde(default)]
    pub tail_keep: Option<usize>,
    #[serde(default = "default_true")]
    pub cache: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RetrieveCacheInput {
    #[schemars(length(max = 256))]
    pub cache_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LcmAppendInput {
    #[schemars(length(max = 256))]
    pub conversation_id: String,
    #[schemars(length(max = 64))]
    pub role: String,
    #[schemars(length(max = 1048576))]
    pub content: String,
    #[serde(default)]
    pub soft_threshold: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LcmNodeInput {
    #[schemars(length(max = 256))]
    pub node_id: String,
}

fn compile_schema<T: JsonSchema>() -> Result<JSONSchema> {
    let json = serde_json::to_value(schemars::schema_for!(T))
        .map_err(|e| anyhow!("schema serialization failed: {e}"))?;
    JSONSchema::options()
        .compile(&json)
        .map_err(|e| anyhow!("schema compile failed: {e}"))
}

fn validate_and_deserialize<T>(value: &Value, schema: &JSONSchema) -> Result<T>
where
    T: DeserializeOwned,
{
    if let Err(errors) = schema.validate(value) {
        let messages: Vec<String> = errors.map(|e| format!("{e}")).collect();
        return Err(anyhow!("schema validation failed: {}", messages.join("; ")));
    }
    serde_json::from_value(value.clone()).map_err(|e| anyhow!("deserialize failed: {e}"))
}

macro_rules! impl_tool_validator {
    ($fn_name:ident, $ty:ty, $static:ident) => {
        static $static: OnceCell<JSONSchema> = OnceCell::new();

        pub fn $fn_name(value: &Value) -> Result<$ty> {
            let schema = $static.get_or_try_init(|| compile_schema::<$ty>())?;
            validate_and_deserialize(value, schema)
        }
    };
}

impl_tool_validator!(validate_lock_mask_input, LockMaskInput, LOCK_MASK_SCHEMA);
impl_tool_validator!(
    validate_compress_array_input,
    CompressArrayInput,
    COMPRESS_ARRAY_SCHEMA
);
impl_tool_validator!(
    validate_retrieve_cache_input,
    RetrieveCacheInput,
    RETRIEVE_CACHE_SCHEMA
);
impl_tool_validator!(validate_lcm_append_input, LcmAppendInput, LCM_APPEND_SCHEMA);
impl_tool_validator!(validate_lcm_node_input, LcmNodeInput, LCM_NODE_SCHEMA);

fn check_string_len(field: &str, s: &str, max: usize) -> Result<()> {
    if s.len() > max {
        return Err(anyhow!("{field} exceeds max length {max}"));
    }
    Ok(())
}

/// Extra semantic bounds after jsonschema (keyword item lengths, array size).
pub fn validate_lock_mask_input_strict(value: &Value) -> Result<LockMaskInput> {
    let input = validate_lock_mask_input(value)?;
    check_string_len("text", &input.text, MAX_TEXT_LEN)?;
    if input.keywords.len() > MAX_KEYWORDS {
        return Err(anyhow!("keywords exceeds max count {MAX_KEYWORDS}"));
    }
    for (i, kw) in input.keywords.iter().enumerate() {
        check_string_len(&format!("keywords[{i}]"), kw, MAX_KEYWORD_ITEM_LEN)?;
    }
    Ok(input)
}

pub fn validate_compress_array_input_strict(value: &Value) -> Result<CompressArrayInput> {
    let input = validate_compress_array_input(value)?;
    if let Value::Array(arr) = &input.value {
        if arr.len() > MAX_JSON_ARRAY_ITEMS {
            return Err(anyhow!(
                "value array exceeds max items {MAX_JSON_ARRAY_ITEMS}"
            ));
        }
    }
    Ok(input)
}

pub fn validate_retrieve_cache_input_strict(value: &Value) -> Result<RetrieveCacheInput> {
    let input = validate_retrieve_cache_input(value)?;
    check_string_len("cache_id", &input.cache_id, MAX_ID_LEN)?;
    Ok(input)
}

pub fn validate_lcm_append_input_strict(value: &Value) -> Result<LcmAppendInput> {
    let input = validate_lcm_append_input(value)?;
    check_string_len("conversation_id", &input.conversation_id, MAX_ID_LEN)?;
    check_string_len("content", &input.content, MAX_CONTENT_LEN)?;
    Ok(input)
}

pub fn validate_lcm_node_input_strict(value: &Value) -> Result<LcmNodeInput> {
    let input = validate_lcm_node_input(value)?;
    check_string_len("node_id", &input.node_id, MAX_ID_LEN)?;
    Ok(input)
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
        let parsed = validate_lock_mask_input_strict(&v).unwrap();
        assert_eq!(parsed.language, "json");
        assert_eq!(parsed.keywords.len(), 2);
    }

    #[test]
    fn accepts_input_without_keywords() {
        let v = json!({"text": "hello", "language": "python"});
        let parsed = validate_lock_mask_input_strict(&v).unwrap();
        assert!(parsed.keywords.is_empty());
    }

    #[test]
    fn rejects_missing_text() {
        let v = json!({"language": "json"});
        assert!(validate_lock_mask_input_strict(&v).is_err());
    }

    #[test]
    fn rejects_wrong_type() {
        let v = json!({"text": 123, "language": "json"});
        assert!(validate_lock_mask_input_strict(&v).is_err());
    }

    #[test]
    fn rejects_oversized_keywords_array() {
        let keywords: Vec<String> = (0..MAX_KEYWORDS + 1).map(|i| format!("k{i}")).collect();
        let v = json!({"text": "x", "language": "json", "keywords": keywords});
        assert!(validate_lock_mask_input_strict(&v).is_err());
    }

    #[test]
    fn compress_array_rejects_huge_array() {
        let arr: Vec<usize> = (0..MAX_JSON_ARRAY_ITEMS + 1).collect();
        let v = json!({"value": arr});
        assert!(validate_compress_array_input_strict(&v).is_err());
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
