use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::DbHandle;
use crate::tokenizer;

pub const DEFAULT_SOFT_THRESHOLD: u64 = 80_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRow {
    pub id: i64,
    pub conversation_id: String,
    pub turn_index: i64,
    pub role: String,
    pub content: String,
    pub tokens: i64,
    pub created_at: i64,
    pub archived_to: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AppendResult {
    pub turn_id: i64,
    pub turn_index: i64,
    pub tokens: i64,
    pub archived_node_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub node_id: String,
    pub conversation_id: String,
    pub depth: i64,
    pub child_count: usize,
    pub total_tokens: i64,
    pub roles: Vec<String>,
    pub created_at: i64,
}

/// Typed error for "node not found" / "cache miss" style lookups.
#[derive(Debug, thiserror::Error)]
#[error("lcm_not_found: no entry for node_id {0}")]
pub struct NotFound(pub String);

/// Appends a turn to the active conversation. Token count comes from the M1
/// tokenizer for parity with downstream compression.
pub fn append(
    conversation_id: &str,
    role: &str,
    content: &str,
    db: &DbHandle,
) -> Result<MessageRow> {
    let (token_ids, _) = tokenizer::token_spans(content)?;
    let token_count = token_ids.len() as i64;
    let now = unix_now()?;
    let conversation_id = conversation_id.to_string();
    let role = role.to_string();
    let content = content.to_string();
    db.call(move |conn| append_on_conn(conn, conversation_id, role, content, token_count, now))
}

/// Appends a turn and archives in the same database-worker turn. This is the
/// deterministic MCP path: the caller receives the archive node created for the
/// append without racing a background worker.
pub fn append_and_maybe_archive(
    conversation_id: &str,
    role: &str,
    content: &str,
    soft_threshold: u64,
    db: &DbHandle,
) -> Result<AppendResult> {
    let (token_ids, _) = tokenizer::token_spans(content)?;
    let token_count = token_ids.len() as i64;
    let now = unix_now()?;
    let conversation_id = conversation_id.to_string();
    let role = role.to_string();
    let content = content.to_string();
    db.call(move |conn| {
        let row = append_on_conn(
            conn,
            conversation_id.clone(),
            role,
            content,
            token_count,
            now,
        )?;
        let archived_node_id = maybe_archive_on_conn(conn, &conversation_id, soft_threshold)?;
        Ok(AppendResult {
            turn_id: row.id,
            turn_index: row.turn_index,
            tokens: row.tokens,
            archived_node_id,
        })
    })
}

/// Sum of tokens for unarchived messages in the conversation.
pub fn active_token_count(conversation_id: &str, db: &DbHandle) -> Result<u64> {
    let conversation_id = conversation_id.to_string();
    db.call(move |conn| active_token_count_on_conn(conn, &conversation_id))
}

/// If the conversation's active token count exceeds `soft_threshold`, archive
/// the oldest unarchived turns into a fresh Depth-0 summary node. The "single
/// huge message" edge case is intentionally NOT archived — we never empty the
/// conversation to under the threshold by archiving the only active turn.
///
/// Returns the new node_id when an archive happened, else `Ok(None)`.
pub fn maybe_archive(
    conversation_id: &str,
    soft_threshold: u64,
    db: &DbHandle,
) -> Result<Option<String>> {
    let conversation_id = conversation_id.to_string();
    db.call(move |conn| maybe_archive_on_conn(conn, &conversation_id, soft_threshold))
}

pub fn describe(node_id: &str, db: &DbHandle) -> Result<NodeMetadata> {
    let node_id = node_id.to_string();
    db.call(move |conn| describe_on_conn(conn, &node_id))
}

pub fn expand(node_id: &str, db: &DbHandle) -> Result<Vec<MessageRow>> {
    let node_id = node_id.to_string();
    db.call(move |conn| expand_on_conn(conn, &node_id))
}

fn append_on_conn(
    conn: &mut Connection,
    conversation_id: String,
    role: String,
    content: String,
    token_count: i64,
    now: i64,
) -> Result<MessageRow> {
    let tx = conn.transaction()?;
    let next_idx: i64 = tx
        .query_row(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM lcm_messages WHERE conversation_id = ?1",
            params![conversation_id],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0);

    tx.execute(
        "INSERT INTO lcm_messages (conversation_id, turn_index, role, content, tokens, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![conversation_id, next_idx, role, content, token_count, now],
    )?;
    let id = tx.last_insert_rowid();
    tx.commit()?;

    Ok(MessageRow {
        id,
        conversation_id,
        turn_index: next_idx,
        role,
        content,
        tokens: token_count,
        created_at: now,
        archived_to: None,
    })
}

fn active_token_count_on_conn(conn: &Connection, conversation_id: &str) -> Result<u64> {
    let n: i64 = conn.query_row(
        "SELECT COALESCE(SUM(tokens), 0) FROM lcm_messages
         WHERE conversation_id = ?1 AND archived_to IS NULL",
        params![conversation_id],
        |r| r.get(0),
    )?;
    Ok(n.max(0) as u64)
}

fn maybe_archive_on_conn(
    conn: &mut Connection,
    conversation_id: &str,
    soft_threshold: u64,
) -> Result<Option<String>> {
    let total: i64 = conn.query_row(
        "SELECT COALESCE(SUM(tokens), 0) FROM lcm_messages
         WHERE conversation_id = ?1 AND archived_to IS NULL",
        params![conversation_id],
        |r| r.get(0),
    )?;
    if (total as u64) <= soft_threshold {
        return Ok(None);
    }

    // Collect active rows oldest-first.
    let rows: Vec<MessageRow> = {
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, turn_index, role, content, tokens, created_at, archived_to
             FROM lcm_messages
             WHERE conversation_id = ?1 AND archived_to IS NULL
             ORDER BY turn_index ASC",
        )?;
        let iter = stmt.query_map(params![conversation_id], |r| {
            Ok(MessageRow {
                id: r.get(0)?,
                conversation_id: r.get(1)?,
                turn_index: r.get(2)?,
                role: r.get(3)?,
                content: r.get(4)?,
                tokens: r.get(5)?,
                created_at: r.get(6)?,
                archived_to: r.get(7)?,
            })
        })?;
        iter.collect::<Result<_, _>>()?
    };

    if rows.len() <= 1 {
        // Never archive the only active turn — that would empty the conversation.
        return Ok(None);
    }

    // Archive oldest rows until remaining active <= soft_threshold, but always
    // leave at least one row active.
    let mut to_archive: Vec<&MessageRow> = Vec::new();
    let mut remaining = total as u64;
    let mut leftover_index = 0;
    for (i, row) in rows.iter().enumerate() {
        if i == rows.len() - 1 {
            // Don't archive the latest turn.
            break;
        }
        to_archive.push(row);
        remaining = remaining.saturating_sub(row.tokens as u64);
        leftover_index = i + 1;
        if remaining <= soft_threshold {
            break;
        }
    }
    let _ = leftover_index;
    if to_archive.is_empty() {
        return Ok(None);
    }

    let node_id = uuid::Uuid::new_v4().to_string();
    let now = unix_now()?;
    let total_archived_tokens: i64 = to_archive.iter().map(|r| r.tokens).sum();
    let child_turn_ids: String =
        serde_json::to_string(&to_archive.iter().map(|r| r.id).collect::<Vec<_>>())?;
    let summary_text = format!(
        "depth-0 archive of {} turns ({} tokens, oldest turn_index {})",
        to_archive.len(),
        total_archived_tokens,
        to_archive.first().unwrap().turn_index,
    );

    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO lcm_summary_nodes
            (id, conversation_id, depth, child_turn_ids, summary_text, total_tokens, created_at)
            VALUES (?1, ?2, 0, ?3, ?4, ?5, ?6)",
        params![
            node_id,
            conversation_id,
            child_turn_ids,
            summary_text,
            total_archived_tokens,
            now
        ],
    )?;
    for row in &to_archive {
        tx.execute(
            "UPDATE lcm_messages SET archived_to = ?1 WHERE id = ?2",
            params![node_id, row.id],
        )?;
    }
    tx.commit()?;

    Ok(Some(node_id))
}

fn describe_on_conn(conn: &Connection, node_id: &str) -> Result<NodeMetadata> {
    let row = conn
        .query_row(
            "SELECT id, conversation_id, depth, total_tokens, created_at
             FROM lcm_summary_nodes WHERE id = ?1",
            params![node_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| NotFound(node_id.to_string()))?;

    let (id, conversation_id, depth, total_tokens, created_at) = row;

    let mut stmt = conn
        .prepare("SELECT role FROM lcm_messages WHERE archived_to = ?1 ORDER BY turn_index ASC")?;
    let roles: Vec<String> = stmt
        .query_map(params![node_id], |r| r.get::<_, String>(0))?
        .collect::<Result<_, _>>()?;
    let child_count = roles.len();

    Ok(NodeMetadata {
        node_id: id,
        conversation_id,
        depth,
        child_count,
        total_tokens,
        roles,
        created_at,
    })
}

fn expand_on_conn(conn: &Connection, node_id: &str) -> Result<Vec<MessageRow>> {
    // First confirm the node exists so we return NotFound, not just an empty vec.
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM lcm_summary_nodes WHERE id = ?1",
            params![node_id],
            |_| Ok(true),
        )
        .optional()?
        .unwrap_or(false);
    if !exists {
        return Err(NotFound(node_id.to_string()).into());
    }
    let mut stmt = conn.prepare(
        "SELECT id, conversation_id, turn_index, role, content, tokens, created_at, archived_to
         FROM lcm_messages
         WHERE archived_to = ?1
         ORDER BY turn_index ASC",
    )?;
    let iter = stmt.query_map(params![node_id], |r| {
        Ok(MessageRow {
            id: r.get(0)?,
            conversation_id: r.get(1)?,
            turn_index: r.get(2)?,
            role: r.get(3)?,
            content: r.get(4)?,
            tokens: r.get(5)?,
            created_at: r.get(6)?,
            archived_to: r.get(7)?,
        })
    })?;
    iter.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn unix_now() -> Result<i64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    /// Deterministic ~filler-tokens string. We tokenize it once with M1's
    /// tokenizer and verify it produces near the target count.
    fn filler(target_tokens: usize) -> String {
        // Each "lorem ipsum word " ≈ 2-3 cl100k tokens; pick a count that gets
        // us into the ballpark.
        let unit = "lorem ipsum dolor sit amet consectetur adipiscing elit ";
        let mut s = String::new();
        while {
            let (ids, _) = tokenizer::token_spans(&s).unwrap();
            ids.len() < target_tokens
        } {
            s.push_str(unit);
        }
        s
    }

    #[test]
    fn append_increments_turn_index() {
        let pool = db::test_pool().unwrap();
        let a = append("conv-1", "user", "hello", &pool).unwrap();
        let b = append("conv-1", "assistant", "hi", &pool).unwrap();
        assert_eq!(a.turn_index, 0);
        assert_eq!(b.turn_index, 1);
        assert!(a.tokens > 0);
    }

    #[test]
    fn active_token_count_sums_unarchived() {
        let pool = db::test_pool().unwrap();
        append("c", "user", "one two three", &pool).unwrap();
        append("c", "user", "four five six", &pool).unwrap();
        let n = active_token_count("c", &pool).unwrap();
        assert!(n > 0);
    }

    #[test]
    fn maybe_archive_below_threshold_is_noop() {
        let pool = db::test_pool().unwrap();
        append("c", "user", "small", &pool).unwrap();
        assert!(maybe_archive("c", 10_000, &pool).unwrap().is_none());
    }

    #[test]
    fn maybe_archive_single_huge_message_does_not_archive() {
        let pool = db::test_pool().unwrap();
        let big = filler(500);
        append("c", "user", &big, &pool).unwrap();
        // soft_threshold lower than the single message — we must NOT archive
        // the only active turn.
        assert!(maybe_archive("c", 100, &pool).unwrap().is_none());
    }

    #[test]
    fn maybe_archive_trims_oldest_until_under_threshold() {
        let pool = db::test_pool().unwrap();
        let chunk = filler(150);
        for _ in 0..8 {
            append("c", "user", &chunk, &pool).unwrap();
        }
        let node_id = maybe_archive("c", 300, &pool).unwrap().expect("archived");
        let active = active_token_count("c", &pool).unwrap();
        assert!(
            active <= 600,
            "active should be near or below threshold after archive; got {active}"
        );

        let meta = describe(&node_id, &pool).unwrap();
        assert!(meta.child_count >= 1);
        assert_eq!(meta.depth, 0);
        assert!(meta.total_tokens > 0);

        let rows = expand(&node_id, &pool).unwrap();
        assert_eq!(rows.len(), meta.child_count);
        for row in &rows {
            assert_eq!(row.content, chunk);
        }
    }

    #[test]
    fn describe_unknown_returns_not_found() {
        let pool = db::test_pool().unwrap();
        let err = describe("nope", &pool).unwrap_err();
        assert!(err.is::<NotFound>(), "got: {err}");
    }

    #[test]
    fn expand_unknown_returns_not_found() {
        let pool = db::test_pool().unwrap();
        let err = expand("nope", &pool).unwrap_err();
        assert!(err.is::<NotFound>(), "got: {err}");
    }

    #[test]
    fn append_and_archive_is_single_db_worker_operation() {
        let pool = db::test_pool().unwrap();
        let chunk = filler(150);
        let mut archived = None;
        for _ in 0..5 {
            let result = append_and_maybe_archive("c", "user", &chunk, 300, &pool).unwrap();
            archived = archived.or(result.archived_node_id);
        }
        assert!(archived.is_some());
        let active = active_token_count("c", &pool).unwrap();
        assert!(active <= 600, "active count after archive: {active}");
    }

    #[test]
    fn concurrent_expand_and_archive_share_db_actor_safely() {
        let pool = db::test_pool().unwrap();
        let chunk = filler(150);
        for _ in 0..8 {
            append("c", "user", &chunk, &pool).unwrap();
        }
        let node_id = maybe_archive("c", 300, &pool).unwrap().expect("archived");

        let expand_pool = pool.clone();
        let expand_node = node_id.clone();
        let reader = std::thread::spawn(move || {
            for _ in 0..20 {
                let rows = expand(&expand_node, &expand_pool).unwrap();
                assert!(!rows.is_empty());
            }
        });

        for _ in 0..10 {
            append_and_maybe_archive("c", "assistant", &chunk, 300, &pool).unwrap();
        }

        reader.join().unwrap();
        let rows = expand(&node_id, &pool).unwrap();
        assert!(!rows.is_empty());
    }
}
