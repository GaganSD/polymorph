use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Sender};
use rusqlite::Connection;
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::thread;

type DbReply = Result<Box<dyn Any + Send>>;
type DbTask = Box<dyn FnOnce(&mut Connection) -> DbReply + Send + 'static>;

struct DbJob {
    task: DbTask,
    reply: Sender<DbReply>,
}

/// Cloneable handle to the single SQLite owner thread.
///
/// `rusqlite::Connection` is synchronous and not `Sync`; callers never touch it
/// directly. Each operation sends an owned closure to the worker and waits for a
/// typed reply, giving us strict thread ownership without `Arc<Mutex<Connection>>`.
#[derive(Clone)]
pub struct DbHandle {
    tx: Sender<DbJob>,
}

impl DbHandle {
    pub fn call<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let (reply_tx, reply_rx) = bounded::<DbReply>(1);
        let task: DbTask =
            Box::new(move |conn| f(conn).map(|value| Box::new(value) as Box<dyn Any + Send>));
        self.tx
            .send(DbJob {
                task,
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("db worker stopped"))?;
        let boxed = reply_rx
            .recv()
            .map_err(|_| anyhow!("db worker dropped response"))??;
        boxed
            .downcast::<T>()
            .map(|value| *value)
            .map_err(|_| anyhow!("db worker returned unexpected response type"))
    }
}

/// Resolves the database path. Order: `$POLYMORPH_DB_PATH`, then
/// `~/.polymorph/cache.db`. Creates the parent directory if missing.
pub fn default_path() -> Result<PathBuf> {
    if let Ok(env) = std::env::var("POLYMORPH_DB_PATH") {
        return Ok(crate::expand_home_path(&env));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home.join(".polymorph").join("cache.db"))
}

/// Opens a connection at `path`, ensures the parent dir exists, applies WAL
/// + foreign_keys pragmas, and runs migrations idempotently.
pub fn open_pool(path: &Path) -> Result<DbHandle> {
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
    Ok(spawn_worker(conn))
}

/// In-memory pool for tests. Same migrations, same pragmas (WAL is a no-op on
/// :memory: but harmless).
pub fn test_pool() -> Result<DbHandle> {
    let conn = Connection::open_in_memory()?;
    // WAL is unsupported in-memory; only set foreign_keys + synchronous.
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;")?;
    migrate(&conn)?;
    Ok(spawn_worker(conn))
}

fn spawn_worker(mut conn: Connection) -> DbHandle {
    let (tx, rx) = bounded::<DbJob>(256);
    thread::Builder::new()
        .name("polymorph-db-worker".into())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                let result = match catch_unwind(AssertUnwindSafe(|| (job.task)(&mut conn))) {
                    Ok(reply) => reply,
                    Err(_) => Err(anyhow!("db worker task panicked")),
                };
                let _ = job.reply.send(result);
            }
        })
        .expect("spawn db worker");
    DbHandle { tx }
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
    use rusqlite::params;
    use std::thread;

    #[test]
    fn test_pool_creates_all_tables() {
        let db = test_pool().unwrap();
        let names: Vec<String> = db
            .call(|conn| {
                conn.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(Into::into)
            })
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
        assert_eq!(
            default_path().unwrap(),
            PathBuf::from("/tmp/some/specific/db")
        );
        match prev {
            Some(v) => std::env::set_var("POLYMORPH_DB_PATH", v),
            None => std::env::remove_var("POLYMORPH_DB_PATH"),
        }
    }

    #[test]
    fn default_path_expands_home_env_var() {
        let prev = std::env::var("POLYMORPH_DB_PATH").ok();
        std::env::set_var("POLYMORPH_DB_PATH", "~/.polymorph/test-cache.db");
        let home = dirs::home_dir().expect("home dir");
        assert_eq!(
            default_path().unwrap(),
            home.join(".polymorph").join("test-cache.db")
        );
        match prev {
            Some(v) => std::env::set_var("POLYMORPH_DB_PATH", v),
            None => std::env::remove_var("POLYMORPH_DB_PATH"),
        }
    }

    #[test]
    fn migrate_is_idempotent() {
        let db = test_pool().unwrap();
        // Running again on the same connection must not error.
        db.call(|conn| {
            migrate(conn)?;
            migrate(conn)
        })
        .unwrap();
    }

    #[test]
    fn cloned_handles_serialize_writes_on_worker_thread() {
        let db = test_pool().unwrap();
        let mut workers = Vec::new();
        for i in 0..8 {
            let db = db.clone();
            workers.push(thread::spawn(move || {
                db.call(move |conn| {
                    conn.execute(
                        "INSERT INTO ccr_cache (id, payload, omitted_count, created_at)
                         VALUES (?1, ?2, 0, 0)",
                        params![format!("cache-{i}"), Vec::<u8>::new()],
                    )?;
                    Ok(())
                })
                .unwrap();
            }));
        }

        for worker in workers {
            worker.join().unwrap();
        }

        let count: i64 = db
            .call(|conn| {
                conn.query_row("SELECT COUNT(*) FROM ccr_cache", [], |row| row.get(0))
                    .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(count, 8);
    }
}
