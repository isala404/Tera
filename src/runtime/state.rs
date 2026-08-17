use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

pub const INIT_RUNTIME_SCHEMA_SQL: &str = r#"
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS daemon_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS main_thread_state (
    id                            INTEGER PRIMARY KEY CHECK (id = 1),
    thread_id                     TEXT NOT NULL,
    turn_id                       TEXT,
    started_at_ms                 INTEGER NOT NULL,
    last_activity_at_ms           INTEGER NOT NULL,
    estimated_cache_warm_until_ms INTEGER NOT NULL,
    model_id                      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_observations (
    model_id        TEXT PRIMARY KEY,
    display_name    TEXT,
    is_default      INTEGER NOT NULL,
    observed_at_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS schedules (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    schedule_type      TEXT NOT NULL,
    one_shot_at_ms     INTEGER,
    dtstart_local      TEXT,
    rrule              TEXT,
    timezone           TEXT NOT NULL,
    task_path          TEXT NOT NULL,
    status             TEXT NOT NULL,
    next_run_at_ms     INTEGER,
    created_at_ms      INTEGER NOT NULL,
    cancelled_at_ms    INTEGER,
    tier               TEXT
);

CREATE TABLE IF NOT EXISTS schedule_runs (
    id                 TEXT PRIMARY KEY,
    schedule_id        TEXT NOT NULL,
    scheduled_for_ms   INTEGER NOT NULL,
    started_at_ms      INTEGER,
    finished_at_ms     INTEGER,
    state              TEXT NOT NULL,
    codex_thread_id    TEXT,
    error              TEXT
);

CREATE TABLE IF NOT EXISTS maintenance_runs (
    id                 TEXT PRIMARY KEY,
    kind               TEXT NOT NULL,
    started_at_ms      INTEGER NOT NULL,
    finished_at_ms     INTEGER,
    state              TEXT NOT NULL,
    new_generation     TEXT,
    error              TEXT
);
"#;

/// Add a column to a table that already exists in a live workspace.
///
/// `CREATE TABLE IF NOT EXISTS` never revisits a table it did not create, so a
/// new column in the schema above reaches a fresh database and no other. SQLite
/// has no `ADD COLUMN IF NOT EXISTS`, so the check is a lookup in `pragma_table_info`.
fn add_column_if_missing(conn: &Connection, table: &str, column: &str, decl: &str) -> Result<()> {
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info(?1) WHERE name = ?2",
        params![table, column],
        |row| row.get(0),
    )?;
    if !exists {
        info!("Adding column {table}.{column} to the runtime database");
        conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainThreadState {
    pub thread_id: String,
    pub turn_id: Option<String>,
    pub started_at_ms: i64,
    pub last_activity_at_ms: i64,
    pub estimated_cache_warm_until_ms: i64,
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObservation {
    pub model_id: String,
    pub display_name: Option<String>,
    pub is_default: bool,
    pub observed_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleItem {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub schedule_type: String, // "once" | "recurring"
    pub one_shot_at_ms: Option<i64>,
    pub dtstart_local: Option<String>,
    pub rrule: Option<String>,
    pub timezone: String,
    pub task_path: String,
    pub status: String, // "active" | "cancelled" | "completed"
    pub next_run_at_ms: Option<i64>,
    pub created_at_ms: i64,
    pub cancelled_at_ms: Option<i64>,
    /// Which model tier runs it: see [`crate::codex::tier`]. Nullable because
    /// schedules created before tiers existed have none; those read back as the
    /// routine tier.
    pub tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleRun {
    pub id: String,
    pub schedule_id: String,
    pub scheduled_for_ms: i64,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
    pub state: String, // "pending" | "running" | "completed" | "failed"
    pub codex_thread_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct RuntimeDb {
    pub conn: Arc<Mutex<Connection>>,
}

impl RuntimeDb {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open runtime state DB at {:?}", db_path))?;
        conn.execute_batch(INIT_RUNTIME_SCHEMA_SQL)?;
        add_column_if_missing(&conn, "schedules", "tier", "TEXT")?;
        info!("Opened runtime state database at {:?}", db_path);
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn set_state_value(&self, key: &str, val: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO daemon_state (key, value) VALUES (?1, ?2)",
            params![key, val],
        )?;
        Ok(())
    }

    pub fn get_state_value(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value FROM daemon_state WHERE key = ?1")?;
        let res = stmt.query_row(params![key], |r| r.get(0)).optional()?;
        Ok(res)
    }

    pub fn save_main_thread(&self, state: &MainThreadState) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO main_thread_state (
                id, thread_id, turn_id, started_at_ms, last_activity_at_ms, estimated_cache_warm_until_ms, model_id
            ) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                state.thread_id,
                state.turn_id,
                state.started_at_ms,
                state.last_activity_at_ms,
                state.estimated_cache_warm_until_ms,
                state.model_id,
            ],
        )?;
        Ok(())
    }

    pub fn get_main_thread(&self) -> Result<Option<MainThreadState>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT thread_id, turn_id, started_at_ms, last_activity_at_ms, estimated_cache_warm_until_ms, model_id
             FROM main_thread_state WHERE id = 1",
        )?;
        let res = stmt
            .query_row([], |row| {
                Ok(MainThreadState {
                    thread_id: row.get(0)?,
                    turn_id: row.get(1)?,
                    started_at_ms: row.get(2)?,
                    last_activity_at_ms: row.get(3)?,
                    estimated_cache_warm_until_ms: row.get(4)?,
                    model_id: row.get(5)?,
                })
            })
            .optional()?;
        Ok(res)
    }

    pub fn record_model_observation(&self, obs: &ModelObservation) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO model_observations (
                model_id, display_name, is_default, observed_at_ms
            ) VALUES (?1, ?2, ?3, ?4)",
            params![
                obs.model_id,
                obs.display_name,
                if obs.is_default { 1 } else { 0 },
                obs.observed_at_ms
            ],
        )?;
        Ok(())
    }

    pub fn get_last_default_model(&self) -> Result<Option<ModelObservation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT model_id, display_name, is_default, observed_at_ms
             FROM model_observations WHERE is_default = 1 ORDER BY observed_at_ms DESC LIMIT 1",
        )?;
        let res = stmt
            .query_row([], |row| {
                let is_def_num: i32 = row.get(2)?;
                Ok(ModelObservation {
                    model_id: row.get(0)?,
                    display_name: row.get(1)?,
                    is_default: is_def_num != 0,
                    observed_at_ms: row.get(3)?,
                })
            })
            .optional()?;
        Ok(res)
    }
}
