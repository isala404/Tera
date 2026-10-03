use crate::runtime::RuntimeDb;
use crate::scheduler::recurrence::ScheduleTiming;
use anyhow::Result;
use chrono::{Local, Utc};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The scheduler's two tables, applied by [`RuntimeDb::open`] alongside the
/// daemon's own. They live here so the shape of a row is next to the queries
/// that read it.
const INIT_SCHEDULER_SCHEMA_SQL: &str = r#"
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
    cancelled_at_ms    INTEGER
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
"#;

pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(INIT_SCHEDULER_SCHEMA_SQL)?;
    Ok(())
}

const SCHEDULE_COLUMNS: &str =
    "id, name, prompt, schedule_type, one_shot_at_ms, rrule, task_path, status, next_run_at_ms";

const RUN_COLUMNS: &str =
    "id, schedule_id, scheduled_for_ms, started_at_ms, finished_at_ms, state, codex_thread_id, error";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleStatus {
    Active,
    Cancelled,
    Completed,
}

impl ScheduleStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Cancelled => "cancelled",
            Self::Completed => "completed",
        }
    }
}

impl std::fmt::Display for ScheduleStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ToSql for ScheduleStatus {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for ScheduleStatus {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_str()? {
            "active" => Ok(Self::Active),
            "cancelled" => Ok(Self::Cancelled),
            "completed" => Ok(Self::Completed),
            other => Err(FromSqlError::Other(
                format!("unknown schedule status {other:?}").into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Running,
    Completed,
    Failed,
}

impl RunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl std::fmt::Display for RunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ToSql for RunState {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for RunState {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_str()? {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(FromSqlError::Other(
                format!("unknown run state {other:?}").into(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScheduleItem {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub timing: ScheduleTiming,
    pub task_path: String,
    pub status: ScheduleStatus,
    pub next_run_at_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ScheduleRun {
    pub id: String,
    pub schedule_id: String,
    pub scheduled_for_ms: i64,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
    pub state: RunState,
    pub codex_thread_id: Option<String>,
    pub error: Option<String>,
}

/// Every query here selects its columns in the same order, and sharing the
/// mapper is what keeps them from drifting when a column is added.
fn run_from_row(row: &Row) -> rusqlite::Result<ScheduleRun> {
    Ok(ScheduleRun {
        id: row.get(0)?,
        schedule_id: row.get(1)?,
        scheduled_for_ms: row.get(2)?,
        started_at_ms: row.get(3)?,
        finished_at_ms: row.get(4)?,
        state: row.get(5)?,
        codex_thread_id: row.get(6)?,
        error: row.get(7)?,
    })
}

fn schedule_from_row(row: &Row) -> rusqlite::Result<ScheduleItem> {
    let kind: String = row.get(3)?;
    let timing = match (kind.as_str(), row.get(4)?, row.get(5)?) {
        ("once", Some(at_ms), _) => ScheduleTiming::Once { at_ms },
        ("recurring", _, Some(rrule)) => ScheduleTiming::Recurring { rrule },
        _ => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                3,
                rusqlite::types::Type::Text,
                format!("invalid schedule timing: type {kind:?} without its time or rule").into(),
            ))
        }
    };
    Ok(ScheduleItem {
        id: row.get(0)?,
        name: row.get(1)?,
        prompt: row.get(2)?,
        timing,
        task_path: row.get(6)?,
        status: row.get(7)?,
        next_run_at_ms: row.get(8)?,
    })
}

pub struct SchedulerDb;

impl SchedulerDb {
    pub fn create_schedule(
        runtime_db: &RuntimeDb,
        name: &str,
        prompt: &str,
        timing: &ScheduleTiming,
        task_path: &str,
    ) -> Result<ScheduleItem> {
        let now_ms = Utc::now().timestamp_millis();
        let item = ScheduleItem {
            id: format!("sched_{}", Uuid::new_v4().simple()),
            name: name.to_string(),
            prompt: prompt.to_string(),
            timing: timing.clone(),
            task_path: task_path.to_string(),
            status: ScheduleStatus::Active,
            next_run_at_ms: timing.next_run(now_ms)?,
        };

        let conn = runtime_db.conn.lock().unwrap();
        // `timezone` is kept for old readers: cron is evaluated in the host's
        // local time, and the offset in force at creation is all it can record.
        conn.execute(
            "INSERT INTO schedules (
                id, name, prompt, schedule_type, one_shot_at_ms, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                item.id,
                item.name,
                item.prompt,
                timing.kind(),
                match timing {
                    ScheduleTiming::Once { at_ms } => Some(*at_ms),
                    ScheduleTiming::Recurring { .. } => None,
                },
                timing.rrule(),
                Local::now().offset().to_string(),
                item.task_path,
                item.status,
                item.next_run_at_ms,
                now_ms,
            ],
        )?;

        Ok(item)
    }

    pub fn list_schedules(runtime_db: &RuntimeDb) -> Result<Vec<ScheduleItem>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query = format!(
            "SELECT {SCHEDULE_COLUMNS} FROM schedules WHERE status = 'active' ORDER BY created_at_ms ASC"
        );
        let mut stmt = conn.prepare(&query)?;

        let rows = stmt.query_map([], schedule_from_row)?;

        let mut items = Vec::new();
        for r in rows {
            items.push(r?);
        }
        Ok(items)
    }

    pub fn get_schedule(runtime_db: &RuntimeDb, schedule_id: &str) -> Result<Option<ScheduleItem>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query = format!("SELECT {SCHEDULE_COLUMNS} FROM schedules WHERE id = ?1");
        let mut stmt = conn.prepare(&query)?;
        Ok(stmt
            .query_row(params![schedule_id], schedule_from_row)
            .optional()?)
    }

    pub fn cancel_schedule(runtime_db: &RuntimeDb, schedule_id: &str) -> Result<bool> {
        let conn = runtime_db.conn.lock().unwrap();
        let now_ms = Utc::now().timestamp_millis();
        let count = conn.execute(
            "UPDATE schedules SET status = 'cancelled', cancelled_at_ms = ?1 WHERE id = ?2 AND status = 'active'",
            params![now_ms, schedule_id],
        )?;
        Ok(count > 0)
    }

    pub fn get_due_schedules(runtime_db: &RuntimeDb, now_ms: i64) -> Result<Vec<ScheduleItem>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query = format!(
            "SELECT {SCHEDULE_COLUMNS} FROM schedules WHERE status = 'active' AND next_run_at_ms IS NOT NULL AND next_run_at_ms <= ?1"
        );
        let mut stmt = conn.prepare(&query)?;

        let rows = stmt.query_map(params![now_ms], schedule_from_row)?;

        let mut items = Vec::new();
        for r in rows {
            items.push(r?);
        }
        Ok(items)
    }

    /// Open a run record. The `schedule_runs` table existed but nothing wrote to
    /// it, so `status` could not tell whether a schedule had ever actually fired.
    pub fn start_run(
        runtime_db: &RuntimeDb,
        schedule_id: &str,
        scheduled_for_ms: i64,
    ) -> Result<String> {
        let id = format!("run_{}", Uuid::new_v4().simple());
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO schedule_runs (id, schedule_id, scheduled_for_ms, started_at_ms, state)
             VALUES (?1, ?2, ?3, ?4, 'running')",
            params![
                id,
                schedule_id,
                scheduled_for_ms,
                Utc::now().timestamp_millis()
            ],
        )?;
        Ok(id)
    }

    pub fn set_run_codex_thread(
        runtime_db: &RuntimeDb,
        run_id: &str,
        codex_thread_id: &str,
    ) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "UPDATE schedule_runs SET codex_thread_id = ?1 WHERE id = ?2",
            params![codex_thread_id, run_id],
        )?;
        Ok(())
    }

    pub fn finish_run(
        runtime_db: &RuntimeDb,
        run_id: &str,
        state: RunState,
        error: Option<&str>,
    ) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "UPDATE schedule_runs SET finished_at_ms = ?1, state = ?2, error = ?3 WHERE id = ?4",
            params![Utc::now().timestamp_millis(), state, error, run_id],
        )?;
        Ok(())
    }

    pub fn recent_runs(runtime_db: &RuntimeDb, limit: usize) -> Result<Vec<ScheduleRun>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query =
            format!("SELECT {RUN_COLUMNS} FROM schedule_runs ORDER BY started_at_ms DESC LIMIT ?1");
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params![limit as i64], run_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn running_runs(runtime_db: &RuntimeDb) -> Result<Vec<ScheduleRun>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query = format!(
            "SELECT {RUN_COLUMNS} FROM schedule_runs WHERE state = 'running' ORDER BY started_at_ms ASC"
        );
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], run_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// How many runs of one schedule have ended failed on one slot. A recovered
    /// run goes back on its slot, so this is how many times that slot crashed.
    pub fn failed_runs_on_slot(
        runtime_db: &RuntimeDb,
        schedule_id: &str,
        scheduled_for_ms: i64,
    ) -> Result<usize> {
        let conn = runtime_db.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM schedule_runs
             WHERE schedule_id = ?1 AND scheduled_for_ms = ?2 AND state = 'failed'",
            params![schedule_id, scheduled_for_ms],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    pub fn get_run(runtime_db: &RuntimeDb, run_id: &str) -> Result<Option<ScheduleRun>> {
        let conn = runtime_db.conn.lock().unwrap();
        let query = format!("SELECT {RUN_COLUMNS} FROM schedule_runs WHERE id = ?1");
        let mut stmt = conn.prepare(&query)?;
        let mut rows = stmt.query_map(params![run_id], run_from_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn set_next_run(
        runtime_db: &RuntimeDb,
        schedule_id: &str,
        next_run_at_ms: Option<i64>,
    ) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "UPDATE schedules SET next_run_at_ms = ?1 WHERE id = ?2",
            params![next_run_at_ms, schedule_id],
        )?;
        Ok(())
    }

    pub fn set_prompt(runtime_db: &RuntimeDb, schedule_id: &str, prompt: &str) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "UPDATE schedules SET prompt = ?1 WHERE id = ?2",
            params![prompt, schedule_id],
        )?;
        Ok(())
    }

    /// Retire a schedule that will not fire again. A cancellation that landed
    /// while its last run was in flight stays a cancellation.
    pub fn complete_schedule(runtime_db: &RuntimeDb, schedule_id: &str) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "UPDATE schedules SET status = 'completed', next_run_at_ms = NULL
             WHERE id = ?1 AND status = 'active'",
            params![schedule_id],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_runtime_db() -> RuntimeDb {
        let dir = tempdir().unwrap();
        let path = dir.path().join("state.sqlite3");
        std::mem::forget(dir);
        RuntimeDb::open(&path).unwrap()
    }

    #[test]
    fn test_unknown_schedule_status_returns_error() {
        let rdb = test_runtime_db();
        let conn = rdb.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO schedules (id, name, prompt, schedule_type, rrule, timezone, task_path, status, created_at_ms)
             VALUES ('s1', 'name', 'prompt', 'recurring', 'EVERY_1H', 'UTC', 'path', 'bogus_status', 1000)",
            [],
        ).unwrap();
        drop(conn);

        let res = SchedulerDb::get_schedule(&rdb, "s1");
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("bogus_status") || err.contains("unknown schedule status"));
    }

    #[test]
    fn test_unknown_run_state_returns_error() {
        let rdb = test_runtime_db();
        let conn = rdb.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO schedule_runs (id, schedule_id, scheduled_for_ms, started_at_ms, state)
             VALUES ('r1', 's1', 1000, 1000, 'exploding_state')",
            [],
        )
        .unwrap();
        drop(conn);

        let res = SchedulerDb::recent_runs(&rdb, 10);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("exploding_state") || err.contains("unknown run state"));
    }

    #[test]
    fn test_malformed_schedule_table_returns_error() {
        let rdb = test_runtime_db();
        let conn = rdb.conn.lock().unwrap();
        conn.execute("DROP TABLE schedules", []).unwrap();
        drop(conn);

        let res = SchedulerDb::list_schedules(&rdb);
        assert!(res.is_err());
    }

    #[test]
    fn test_set_run_codex_thread_persists_durable_mapping() {
        let rdb = test_runtime_db();
        let run_id = SchedulerDb::start_run(&rdb, "sched_test", 1000).unwrap();
        SchedulerDb::set_run_codex_thread(&rdb, &run_id, "th_isolated_42").unwrap();

        let runs = SchedulerDb::recent_runs(&rdb, 1).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run_id);
        assert_eq!(runs[0].codex_thread_id.as_deref(), Some("th_isolated_42"));
        assert_eq!(runs[0].state, RunState::Running);
    }

    #[test]
    fn test_invalid_schedule_timing_returns_error() {
        let rdb = test_runtime_db();
        let conn = rdb.conn.lock().unwrap();
        // schedule_type is neither 'once' nor 'recurring'
        conn.execute(
            "INSERT INTO schedules (id, name, prompt, schedule_type, timezone, task_path, status, created_at_ms)
             VALUES ('s2', 'name', 'prompt', 'cron_special', 'UTC', 'path', 'active', 1000)",
            [],
        ).unwrap();
        drop(conn);

        let res = SchedulerDb::list_schedules(&rdb);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("cron_special") || err.contains("invalid schedule timing"));
    }
}
