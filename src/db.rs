use anyhow::{anyhow, Context, Result};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Single shared rusqlite connection. `Connection` is `!Sync` so the only safe
/// way to share it across threads (MCP request handler + LCM archiver worker)
/// is behind a Mutex. We hold the lock per SQL call, never across other I/O.
pub type Pool = Arc<Mutex<Connection>>;

/// Resolves the database path. Order: `$POLYMORPH_DB_PATH`, then
/// `~/.polymorph/cache.db`. Creates the parent directory if missing.
pub fn default_path() -> Result<PathBuf> {
    if let Ok(env) = std::env::var("POLYMORPH_DB_PATH") {
        return Ok(PathBuf::from(env));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home.join(".polymorph").join("cache.db"))
}

/// Opens a connection at `path`, ensures the parent dir exists, applies WAL
/// + foreign_keys pragmas, and runs migrations idempotently.
pub fn open_pool(path: &Path) -> Result<Pool> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db parent dir {}", parent.display()))?;
        }
    }
    let conn = Connection::open(path)
        .with_context(|| format!("opening sqlite db at {}", path.display()))?;
    configure(&conn)?;
    migrate(&conn)?;
    Ok(Arc::new(Mutex::new(conn)))
}

/// In-memory pool for tests. Same migrations, same pragmas (WAL is a no-op on
/// :memory: but harmless).
pub fn test_pool() -> Result<Pool> {
    let conn = Connection::open_in_memory()?;
    // WAL is unsupported in-memory; only set foreign_keys + synchronous.
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;")?;
    migrate(&conn)?;
    Ok(Arc::new(Mutex::new(conn)))
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=ON;",
    )?;
    Ok(())
}

fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS ccr_cache (
            id            TEXT PRIMARY KEY,
            payload       BLOB NOT NULL,
            omitted_count INTEGER NOT NULL,
            created_at    INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lcm_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            turn_index      INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            tokens          INTEGER NOT NULL,
            created_at      INTEGER NOT NULL,
            archived_to     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_messages_active
            ON lcm_messages(conversation_id, archived_to, turn_index);

        CREATE TABLE IF NOT EXISTS lcm_summary_nodes (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            depth           INTEGER NOT NULL,
            child_turn_ids  TEXT NOT NULL,
            summary_text    TEXT NOT NULL,
            total_tokens    INTEGER NOT NULL,
            created_at      INTEGER NOT NULL
        );
        "#,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creates_all_tables() {
        let pool = test_pool().unwrap();
        let conn = pool.lock().unwrap();
        let names: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(names.contains(&"ccr_cache".to_string()));
        assert!(names.contains(&"lcm_messages".to_string()));
        assert!(names.contains(&"lcm_summary_nodes".to_string()));
    }

    #[test]
    fn open_pool_creates_parent_dir() {
        let tmp = std::env::temp_dir().join(format!("polymorph-test-{}", uuid::Uuid::new_v4()));
        let db_path = tmp.join("nested").join("cache.db");
        let _pool = open_pool(&db_path).unwrap();
        assert!(db_path.exists());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn default_path_honors_env_var() {
        let prev = std::env::var("POLYMORPH_DB_PATH").ok();
        std::env::set_var("POLYMORPH_DB_PATH", "/tmp/some/specific/db");
        assert_eq!(default_path().unwrap(), PathBuf::from("/tmp/some/specific/db"));
        match prev {
            Some(v) => std::env::set_var("POLYMORPH_DB_PATH", v),
            None => std::env::remove_var("POLYMORPH_DB_PATH"),
        }
    }

    #[test]
    fn migrate_is_idempotent() {
        let pool = test_pool().unwrap();
        // Running again on the same connection must not error.
        let conn = pool.lock().unwrap();
        migrate(&conn).unwrap();
        migrate(&conn).unwrap();
    }
}
