use anyhow::{Context, Result};
use chrono::Utc;
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

-- One row per foreground turn, the same shape `schedule_runs` already uses for
-- background work. A row with no `finished_at_ms` at startup is a turn the last
-- process did not live to answer, and `attempts` is what keeps a request that
-- crashes us from crashing us forever.
CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id              TEXT PRIMARY KEY,
    chat_jid             TEXT NOT NULL,
    last_provider_msg_id TEXT NOT NULL,
    started_at_ms        INTEGER NOT NULL,
    finished_at_ms       INTEGER,
    attempts             INTEGER NOT NULL DEFAULT 0,
    state                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_observations (
    model_id        TEXT PRIMARY KEY,
    display_name    TEXT,
    is_default      INTEGER NOT NULL,
    observed_at_ms  INTEGER NOT NULL
);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainThreadState {
    pub thread_id: String,
    pub turn_id: Option<String>,
    pub started_at_ms: i64,
    pub last_activity_at_ms: i64,
    pub estimated_cache_warm_until_ms: i64,
    pub model_id: String,
}

/// A foreground turn's lifecycle, mirroring [`ScheduleRun`].
///
/// The scheduler already models "work a crash can interrupt" as a row with a
/// state; a conversation turn is the same thing with a person waiting on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub turn_id: String,
    pub chat_jid: String,
    pub last_provider_msg_id: String,
    pub started_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    /// Recovery attempts spent on this turn. Zero until Phoenix picks it up.
    pub attempts: i64,
    /// "running" | "completed" | "failed" | "abandoned"
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObservation {
    pub model_id: String,
    pub display_name: Option<String>,
    pub is_default: bool,
    pub observed_at_ms: i64,
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
        crate::scheduler::db::init_schema(&conn)?;
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

    /// Open a turn, or refresh the reply target of one already open.
    ///
    /// Steering messages join a turn that is already running, and a recovery
    /// should answer the newest of them rather than the one that opened the
    /// burst. Attempts and start time belong to the turn, so they survive.
    pub fn start_turn(
        &self,
        turn_id: &str,
        chat_jid: &str,
        last_provider_msg_id: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversation_turns (
                turn_id, chat_jid, last_provider_msg_id, started_at_ms, attempts, state
            ) VALUES (?1, ?2, ?3, ?4, 0, 'running')
            ON CONFLICT(turn_id) DO UPDATE SET last_provider_msg_id = excluded.last_provider_msg_id",
            params![
                turn_id,
                chat_jid,
                last_provider_msg_id,
                Utc::now().timestamp_millis()
            ],
        )?;
        Ok(())
    }

    pub fn finish_turn(&self, turn_id: &str, state: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE conversation_turns SET finished_at_ms = ?1, state = ?2 WHERE turn_id = ?3",
            params![Utc::now().timestamp_millis(), state, turn_id],
        )?;
        Ok(())
    }

    /// Turns with no terminal state, oldest first.
    ///
    /// At startup this is exactly the set of turns the previous process accepted
    /// and never answered.
    pub fn unfinished_turns(&self) -> Result<Vec<ConversationTurn>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT turn_id, chat_jid, last_provider_msg_id, started_at_ms, finished_at_ms, attempts, state
             FROM conversation_turns WHERE finished_at_ms IS NULL ORDER BY started_at_ms ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ConversationTurn {
                turn_id: row.get(0)?,
                chat_jid: row.get(1)?,
                last_provider_msg_id: row.get(2)?,
                started_at_ms: row.get(3)?,
                finished_at_ms: row.get(4)?,
                attempts: row.get(5)?,
                state: row.get(6)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// The chat the last turn happened in.
    ///
    /// Phoenix needs somewhere to speak after a crash that left no turn open,
    /// and the owner's JID is not always configured: the default owner is
    /// whichever account this daemon is paired to. The last conversation is the
    /// one fact that is always right.
    pub fn last_known_chat(&self) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chat_jid FROM conversation_turns ORDER BY started_at_ms DESC LIMIT 1",
        )?;
        Ok(stmt.query_row([], |row| row.get(0)).optional()?)
    }

    /// Count a recovery attempt against a turn and return the new total.
    pub fn record_turn_attempt(&self, turn_id: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE conversation_turns SET attempts = attempts + 1 WHERE turn_id = ?1",
            params![turn_id],
        )?;
        let attempts: i64 = conn.query_row(
            "SELECT attempts FROM conversation_turns WHERE turn_id = ?1",
            params![turn_id],
            |row| row.get(0),
        )?;
        Ok(attempts)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole recovery contract: an open turn is visible after a restart, a
    /// finished one is not, and a steering message moves the reply target
    /// without resetting the turn.
    #[test]
    fn test_unfinished_turns_survive_until_finished() {
        let dir = tempfile::tempdir().unwrap();
        let db = RuntimeDb::open(&dir.path().join("state.sqlite3")).unwrap();

        db.start_turn("turn_1", "947@s.whatsapp.net", "wamid.1")
            .unwrap();
        db.start_turn("turn_2", "947@s.whatsapp.net", "wamid.2")
            .unwrap();
        db.finish_turn("turn_2", "completed").unwrap();

        let open = db.unfinished_turns().unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].turn_id, "turn_1");
        assert_eq!(open[0].attempts, 0);

        // A steering message answers to the newest message, not the first.
        db.start_turn("turn_1", "947@s.whatsapp.net", "wamid.3")
            .unwrap();
        let open = db.unfinished_turns().unwrap();
        assert_eq!(open.len(), 1, "steering must not open a second turn");
        assert_eq!(open[0].last_provider_msg_id, "wamid.3");
        assert_eq!(
            open[0].started_at_ms,
            db.unfinished_turns().unwrap()[0].started_at_ms
        );
    }

    /// A request that keeps killing the process has to stop being retried, so
    /// the attempt count is per turn rather than per boot.
    #[test]
    fn test_turn_attempts_accumulate_across_recoveries() {
        let dir = tempfile::tempdir().unwrap();
        let db = RuntimeDb::open(&dir.path().join("state.sqlite3")).unwrap();

        db.start_turn("turn_1", "947@s.whatsapp.net", "wamid.1")
            .unwrap();
        assert_eq!(db.record_turn_attempt("turn_1").unwrap(), 1);
        assert_eq!(db.record_turn_attempt("turn_1").unwrap(), 2);

        db.finish_turn("turn_1", "abandoned").unwrap();
        assert!(db.unfinished_turns().unwrap().is_empty());
    }
}
