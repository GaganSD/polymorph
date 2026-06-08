use anyhow::Result;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::DbHandle;

#[derive(Debug, Clone, Copy)]
pub struct CcrOpts {
    pub head_keep: usize,
    pub tail_keep: usize,
    pub min_compress_len: usize,
}

impl Default for CcrOpts {
    fn default() -> Self {
        Self {
            head_keep: 3,
            tail_keep: 3,
            min_compress_len: 20,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedArray {
    pub compressed: Value,
    pub cache_id: Option<String>,
    pub omitted_count: usize,
}

/// Compress a JSON array by keeping head + tail elements and stashing the
/// middle in the SQLite cache (when `persist == true`). For non-array inputs
/// or arrays below the compression threshold, returns the value unchanged.
pub fn compress_array(
    value: Value,
    opts: CcrOpts,
    db: &DbHandle,
    persist: bool,
) -> Result<CompressedArray> {
    let arr = match value {
        Value::Array(a) => a,
        other => {
            return Ok(CompressedArray {
                compressed: other,
                cache_id: None,
                omitted_count: 0,
            });
        }
    };

    let len = arr.len();
    let Some(edge) = opts.head_keep.checked_add(opts.tail_keep) else {
        return Ok(CompressedArray {
            compressed: Value::Array(arr),
            cache_id: None,
            omitted_count: 0,
        });
    };
    let Some(compress_at) = edge.checked_add(opts.min_compress_len.max(1)) else {
        return Ok(CompressedArray {
            compressed: Value::Array(arr),
            cache_id: None,
            omitted_count: 0,
        });
    };
    if len < compress_at {
        return Ok(CompressedArray {
            compressed: Value::Array(arr),
            cache_id: None,
            omitted_count: 0,
        });
    }

    debug_assert!(edge < len, "compression requires a non-empty middle slice");
    let head: Vec<Value> = arr.iter().take(opts.head_keep).cloned().collect();
    let middle: Vec<Value> = arr
        .iter()
        .skip(opts.head_keep)
        .take(len - edge)
        .cloned()
        .collect();
    let tail: Vec<Value> = arr.iter().skip(len - opts.tail_keep).cloned().collect();
    let omitted_count = middle.len();
    debug_assert!(omitted_count >= opts.min_compress_len.max(1));
    debug_assert_eq!(head.len(), opts.head_keep);
    debug_assert_eq!(tail.len(), opts.tail_keep);

    let cache_id = if persist {
        let id = uuid::Uuid::new_v4().to_string();
        let payload = serde_json::to_vec(&Value::Array(middle))?;
        let now = unix_now()?;
        let id_for_db = id.clone();
        db.call(move |conn| {
            conn.execute(
                "INSERT INTO ccr_cache (id, payload, omitted_count, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![id_for_db, payload, omitted_count as i64, now],
            )?;
            Ok(())
        })?;
        Some(id)
    } else {
        None
    };

    let summary = json!({
        "__polymorph_cache_id": cache_id,
        "__omitted_count": omitted_count,
        "__summary": format!(
            "{} items elided{}",
            omitted_count,
            if cache_id.is_some() {
                "; call polymorph_retrieve_cache with the id to expand"
            } else {
                " (cache:false, not retrievable)"
            }
        ),
    });

    let mut compressed: Vec<Value> = Vec::with_capacity(head.len() + 1 + tail.len());
    compressed.extend(head);
    compressed.push(summary);
    compressed.extend(tail);
    debug_assert_eq!(compressed.len(), opts.head_keep + 1 + opts.tail_keep);

    Ok(CompressedArray {
        compressed: Value::Array(compressed),
        cache_id,
        omitted_count,
    })
}

/// Stash an opaque JSON payload in the cache and return its id. Reuses the
/// `ccr_cache` table (and so the existing `polymorph_retrieve_cache` tool) so
/// `compress_log` can make the *original* log text retrievable after pruning —
/// the reversibility guarantee for an audit-log compressor.
pub fn stash(value: &Value, db: &DbHandle) -> Result<String> {
    let id = uuid::Uuid::new_v4().to_string();
    let payload = serde_json::to_vec(value)?;
    let now = unix_now()?;
    let id_for_db = id.clone();
    db.call(move |conn| {
        conn.execute(
            "INSERT INTO ccr_cache (id, payload, omitted_count, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id_for_db, payload, 0i64, now],
        )?;
        Ok(())
    })?;
    Ok(id)
}

/// Retrieves the omitted middle slice for a given cache id. Returns a typed
/// `CacheMiss` error when the id isn't found so the MCP layer can map to a
/// structured client-side error.
pub fn retrieve(cache_id: &str, db: &DbHandle) -> Result<Value> {
    let cache_id = cache_id.to_string();
    let row: Option<Vec<u8>> = db.call({
        let cache_id = cache_id.clone();
        move |conn| {
            conn.query_row(
                "SELECT payload FROM ccr_cache WHERE id = ?1",
                params![cache_id],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(Into::into)
        }
    })?;
    match row {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(CacheMiss(cache_id.to_string()).into()),
    }
}

#[derive(Debug, thiserror::Error)]
#[error("cache_miss: no entry for cache_id {0}")]
pub struct CacheMiss(pub String);

fn unix_now() -> Result<i64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn long_array(n: usize) -> Value {
        Value::Array((0..n).map(|i| json!({"i": i})).collect())
    }

    #[test]
    fn compresses_long_array() {
        let pool = db::test_pool().unwrap();
        let res = compress_array(long_array(50), CcrOpts::default(), &pool, true).unwrap();
        let arr = res.compressed.as_array().unwrap();
        assert_eq!(arr.len(), 3 + 1 + 3, "head + summary + tail");
        assert!(res.cache_id.is_some());
        assert_eq!(res.omitted_count, 44);
        assert_eq!(arr[3]["__omitted_count"], 44);
        assert_eq!(arr[3]["__polymorph_cache_id"], res.cache_id.unwrap());
    }

    #[test]
    fn round_trip_recovers_middle() {
        let pool = db::test_pool().unwrap();
        let original = long_array(50);
        let middle: Vec<Value> = original
            .as_array()
            .unwrap()
            .iter()
            .skip(3)
            .take(44)
            .cloned()
            .collect();
        let res = compress_array(original, CcrOpts::default(), &pool, true).unwrap();
        let cache_id = res.cache_id.unwrap();
        let recovered = retrieve(&cache_id, &pool).unwrap();
        assert_eq!(recovered, Value::Array(middle));
    }

    #[test]
    fn short_array_returned_unchanged() {
        let pool = db::test_pool().unwrap();
        let res = compress_array(long_array(5), CcrOpts::default(), &pool, true).unwrap();
        assert!(res.cache_id.is_none());
        assert_eq!(res.compressed.as_array().unwrap().len(), 5);
        assert_eq!(res.omitted_count, 0);
    }

    #[test]
    fn exact_threshold_boundaries_are_stable() {
        let pool = db::test_pool().unwrap();
        let unchanged = compress_array(long_array(25), CcrOpts::default(), &pool, true).unwrap();
        assert!(unchanged.cache_id.is_none());
        assert_eq!(unchanged.compressed.as_array().unwrap().len(), 25);

        let compressed = compress_array(long_array(26), CcrOpts::default(), &pool, true).unwrap();
        assert_eq!(compressed.omitted_count, 20);
        assert_eq!(compressed.compressed.as_array().unwrap().len(), 7);
    }

    #[test]
    fn head_tail_edges_match_original_indices() {
        let pool = db::test_pool().unwrap();
        let original = long_array(50);
        let res = compress_array(original.clone(), CcrOpts::default(), &pool, true).unwrap();
        let src = original.as_array().unwrap();
        let out = res.compressed.as_array().unwrap();

        assert_eq!(&out[0..3], &src[0..3]);
        assert_eq!(&out[4..7], &src[47..50]);
    }

    #[test]
    fn zero_head_or_tail_still_emits_valid_summary_boundary() {
        let pool = db::test_pool().unwrap();
        let no_head = compress_array(
            long_array(30),
            CcrOpts {
                head_keep: 0,
                tail_keep: 3,
                min_compress_len: 20,
            },
            &pool,
            true,
        )
        .unwrap();
        let no_head_arr = no_head.compressed.as_array().unwrap();
        assert_eq!(no_head_arr.len(), 4);
        assert!(no_head_arr[0].get("__summary").is_some());

        let no_tail = compress_array(
            long_array(30),
            CcrOpts {
                head_keep: 3,
                tail_keep: 0,
                min_compress_len: 20,
            },
            &pool,
            true,
        )
        .unwrap();
        let no_tail_arr = no_tail.compressed.as_array().unwrap();
        assert_eq!(no_tail_arr.len(), 4);
        assert!(no_tail_arr[3].get("__summary").is_some());
    }

    #[test]
    fn head_tail_sum_above_len_returns_unchanged() {
        let pool = db::test_pool().unwrap();
        let opts = CcrOpts {
            head_keep: 10,
            tail_keep: 10,
            min_compress_len: 5,
        };
        let res = compress_array(long_array(15), opts, &pool, true).unwrap();
        assert!(res.cache_id.is_none());
        assert_eq!(res.compressed.as_array().unwrap().len(), 15);
    }

    #[test]
    fn non_array_returned_unchanged() {
        let pool = db::test_pool().unwrap();
        let res = compress_array(json!({"key": "value"}), CcrOpts::default(), &pool, true).unwrap();
        assert!(res.cache_id.is_none());
        assert_eq!(res.compressed, json!({"key": "value"}));
    }

    #[test]
    fn cache_false_skips_persistence() {
        let pool = db::test_pool().unwrap();
        let res = compress_array(long_array(50), CcrOpts::default(), &pool, false).unwrap();
        assert!(res.cache_id.is_none());
        let arr = res.compressed.as_array().unwrap();
        assert_eq!(arr.len(), 3 + 1 + 3);
        assert!(arr[3]["__summary"]
            .as_str()
            .unwrap()
            .contains("not retrievable"));
        assert!(arr[3]["__polymorph_cache_id"].is_null());
        assert_eq!(res.omitted_count, 44);
    }

    #[test]
    fn zero_min_compress_len_never_injects_empty_middle_summary() {
        let pool = db::test_pool().unwrap();
        let opts = CcrOpts {
            head_keep: 3,
            tail_keep: 3,
            min_compress_len: 0,
        };
        let res = compress_array(long_array(6), opts, &pool, true).unwrap();
        assert!(res.cache_id.is_none());
        assert_eq!(res.omitted_count, 0);
        assert_eq!(res.compressed.as_array().unwrap().len(), 6);
    }

    #[test]
    fn retrieve_unknown_id_returns_cache_miss() {
        let pool = db::test_pool().unwrap();
        let err = retrieve("not-a-real-id", &pool).unwrap_err();
        assert!(err.is::<CacheMiss>(), "got: {err}");
    }
}
