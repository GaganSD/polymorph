//! Tool input shapes + post-deserialize semantic checks.
//!
//! The protocol layer (rmcp) handles JSON-RPC framing and serde deserialization
//! of the input structs below into typed `Parameters<T>`. This module supplies:
//!
//! 1. The `#[derive(Deserialize, JsonSchema)]` structs that rmcp publishes as
//!    each tool's `inputSchema` and uses to deserialize arguments.
//! 2. The `_strict` validators that enforce semantic bounds (string lengths,
//!    array counts) that the schema alone can't cleanly express. These run
//!    inside each handler after rmcp has produced a `Parameters<T>`.
//!
//! A bounded `AsyncRead` wrapper in `crate::transport` enforces the per-message
//! payload cap below at the byte stream — defeats zip-bomb-style inflation
//! before any JSON parsing.

use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

/// Hard cap on a single MCP message size. JSON-RPC messages above this are
/// rejected at the reader before any parsing.
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LockMaskInput {
    #[schemars(length(max = MAX_TEXT_LEN))]
    pub text: String,
    #[schemars(length(max = 16))]
    pub language: String,
    #[schemars(length(max = MAX_KEYWORDS), inner(length(max = MAX_KEYWORD_ITEM_LEN)))]
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
    #[schemars(length(max = MAX_ID_LEN))]
    pub cache_id: String,
}

/// Max bytes read from a local log file by `compress_log`'s `path` arg. Larger
/// than [`MAX_TEXT_LEN`] (the inline-`text` cap) so 10 MB logs can be fed by path.
pub const MAX_LOG_FILE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PATH_LEN: usize = 4_096;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompressLogInput {
    /// Inline log text (≤ 1 MB). Provide this OR `path`, not both.
    #[schemars(length(max = MAX_TEXT_LEN))]
    #[serde(default)]
    pub text: Option<String>,
    /// Local filesystem path to read the log from — use this for large logs
    /// (≤ 64 MB) that exceed the inline text/payload caps. Provide this OR `text`.
    #[schemars(length(max = MAX_PATH_LEN))]
    #[serde(default)]
    pub path: Option<String>,
    /// Structural-lock grammar: "text" (default, no AST locking — correct for raw
    /// logs), "json", or "python".
    #[schemars(length(max = 16))]
    #[serde(default)]
    pub language: Option<String>,
    /// Extra substrings to force-keep (API keys, resource ids, anything the model
    /// must never drop), locked via the DAAC scanner.
    #[schemars(length(max = MAX_KEYWORDS), inner(length(max = MAX_KEYWORD_ITEM_LEN)))]
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Fraction of unlocked tokens to drop, in (0,1). Omit for the model default.
    #[serde(default)]
    pub target_rate: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LcmAppendInput {
    #[schemars(length(max = MAX_ID_LEN))]
    pub conversation_id: String,
    #[schemars(length(max = 64))]
    pub role: String,
    #[schemars(length(max = MAX_CONTENT_LEN))]
    pub content: String,
    #[serde(default)]
    pub soft_threshold: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LcmNodeInput {
    #[schemars(length(max = MAX_ID_LEN))]
    pub node_id: String,
}

fn check_string_len(field: &str, s: &str, max: usize) -> Result<()> {
    if s.len() > max {
        return Err(anyhow!("{field} exceeds max length {max}"));
    }
    Ok(())
}

pub fn check_lock_mask_input(input: &LockMaskInput) -> Result<()> {
    check_string_len("text", &input.text, MAX_TEXT_LEN)?;
    check_string_len("language", &input.language, 16)?;
    if input.keywords.len() > MAX_KEYWORDS {
        return Err(anyhow!("keywords exceeds max count {MAX_KEYWORDS}"));
    }
    for (i, kw) in input.keywords.iter().enumerate() {
        check_string_len(&format!("keywords[{i}]"), kw, MAX_KEYWORD_ITEM_LEN)?;
    }
    Ok(())
}

pub fn check_compress_array_input(input: &CompressArrayInput) -> Result<()> {
    if input.head_keep.unwrap_or(0) > MAX_JSON_ARRAY_ITEMS {
        return Err(anyhow!(
            "head_keep exceeds max items {MAX_JSON_ARRAY_ITEMS}"
        ));
    }
    if input.tail_keep.unwrap_or(0) > MAX_JSON_ARRAY_ITEMS {
        return Err(anyhow!(
            "tail_keep exceeds max items {MAX_JSON_ARRAY_ITEMS}"
        ));
    }
    let edge_size_ok = input
        .head_keep
        .unwrap_or(0)
        .checked_add(input.tail_keep.unwrap_or(0))
        .map(|edge| edge <= MAX_JSON_ARRAY_ITEMS)
        .unwrap_or(false);
    if !edge_size_ok {
        return Err(anyhow!(
            "head_keep + tail_keep exceeds max items {MAX_JSON_ARRAY_ITEMS}"
        ));
    }
    if let Value::Array(arr) = &input.value {
        if arr.len() > MAX_JSON_ARRAY_ITEMS {
            return Err(anyhow!(
                "value array exceeds max items {MAX_JSON_ARRAY_ITEMS}"
            ));
        }
    }
    Ok(())
}

pub fn check_retrieve_cache_input(input: &RetrieveCacheInput) -> Result<()> {
    check_string_len("cache_id", &input.cache_id, MAX_ID_LEN)
}

pub fn check_compress_log_input(input: &CompressLogInput) -> Result<()> {
    match (&input.text, &input.path) {
        (Some(_), Some(_)) => {
            return Err(anyhow!("provide exactly one of `text` or `path`, not both"))
        }
        (None, None) => return Err(anyhow!("provide one of `text` or `path`")),
        (Some(t), None) => check_string_len("text", t, MAX_TEXT_LEN)?,
        (None, Some(p)) => check_string_len("path", p, MAX_PATH_LEN)?,
    }
    if input.keywords.len() > MAX_KEYWORDS {
        return Err(anyhow!("keywords exceeds max count {MAX_KEYWORDS}"));
    }
    for (i, kw) in input.keywords.iter().enumerate() {
        check_string_len(&format!("keywords[{i}]"), kw, MAX_KEYWORD_ITEM_LEN)?;
    }
    if let Some(r) = input.target_rate {
        if !(r > 0.0 && r < 1.0 && r.is_finite()) {
            return Err(anyhow!("target_rate must be a finite number in (0,1)"));
        }
    }
    if let Some(l) = &input.language {
        if crate::Language::parse(l).is_none() {
            return Err(anyhow!("unsupported language: {l} (use text|json|python)"));
        }
    }
    Ok(())
}

pub fn check_lcm_append_input(input: &LcmAppendInput) -> Result<()> {
    check_string_len("conversation_id", &input.conversation_id, MAX_ID_LEN)?;
    check_string_len("role", &input.role, 64)?;
    check_string_len("content", &input.content, MAX_CONTENT_LEN)
}

pub fn check_lcm_node_input(input: &LcmNodeInput) -> Result<()> {
    check_string_len("node_id", &input.node_id, MAX_ID_LEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse<T: serde::de::DeserializeOwned>(v: serde_json::Value) -> Result<T> {
        serde_json::from_value(v).map_err(|e| anyhow!("{e}"))
    }

    #[test]
    fn accepts_valid_lock_mask() {
        let i: LockMaskInput = parse(serde_json::json!({
            "text": "hello", "language": "json", "keywords": ["k1", "k2"]
        }))
        .unwrap();
        check_lock_mask_input(&i).unwrap();
    }

    #[test]
    fn accepts_lock_mask_without_keywords() {
        let i: LockMaskInput =
            parse(serde_json::json!({"text": "hi", "language": "python"})).unwrap();
        check_lock_mask_input(&i).unwrap();
        assert!(i.keywords.is_empty());
    }

    #[test]
    fn rejects_oversized_keywords_array() {
        let keywords: Vec<String> = (0..MAX_KEYWORDS + 1).map(|i| format!("k{i}")).collect();
        let i = LockMaskInput {
            text: "x".into(),
            language: "json".into(),
            keywords,
        };
        assert!(check_lock_mask_input(&i).is_err());
    }

    #[test]
    fn rejects_oversized_text() {
        let i = LockMaskInput {
            text: "a".repeat(MAX_TEXT_LEN + 1),
            language: "json".into(),
            keywords: vec![],
        };
        assert!(check_lock_mask_input(&i).is_err());
    }

    #[test]
    fn rejects_oversized_keyword_item() {
        let i = LockMaskInput {
            text: "x".into(),
            language: "json".into(),
            keywords: vec!["a".repeat(MAX_KEYWORD_ITEM_LEN + 1)],
        };
        assert!(check_lock_mask_input(&i).is_err());
    }

    #[test]
    fn compress_array_rejects_huge_array() {
        let arr: Vec<usize> = (0..MAX_JSON_ARRAY_ITEMS + 1).collect();
        let i = CompressArrayInput {
            value: serde_json::to_value(arr).unwrap(),
            head_keep: None,
            tail_keep: None,
            cache: true,
        };
        assert!(check_compress_array_input(&i).is_err());
    }

    #[test]
    fn compress_array_rejects_pathological_edges() {
        let i = CompressArrayInput {
            value: serde_json::json!([1, 2, 3]),
            head_keep: Some(MAX_JSON_ARRAY_ITEMS),
            tail_keep: Some(1),
            cache: true,
        };
        assert!(check_compress_array_input(&i).is_err());
    }

    #[test]
    fn retrieve_cache_rejects_oversized_id() {
        let i = RetrieveCacheInput {
            cache_id: "x".repeat(MAX_ID_LEN + 1),
        };
        assert!(check_retrieve_cache_input(&i).is_err());
    }

    #[test]
    fn lcm_append_rejects_oversized_content() {
        let i = LcmAppendInput {
            conversation_id: "c".into(),
            role: "user".into(),
            content: "a".repeat(MAX_CONTENT_LEN + 1),
            soft_threshold: None,
        };
        assert!(check_lcm_append_input(&i).is_err());
    }

    #[test]
    fn lcm_node_rejects_oversized_id() {
        let i = LcmNodeInput {
            node_id: "x".repeat(MAX_ID_LEN + 1),
        };
        assert!(check_lcm_node_input(&i).is_err());
    }
}
