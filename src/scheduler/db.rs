use crate::runtime::RuntimeDb;
use crate::scheduler::recurrence::ScheduleTiming;
use anyhow::Result;
use chrono::{Local, Utc};
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

pub(crate) const SCHEDULE_COLUMNS: &str =
    "id, name, prompt, schedule_type, one_shot_at_ms, dtstart_local, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms, cancelled_at_ms";

pub(crate) const RUN_COLUMNS: &str =
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

impl std::str::FromStr for ScheduleStatus {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "cancelled" => Ok(Self::Cancelled),
            "completed" => Ok(Self::Completed),
            other => Err(anyhow::anyhow!("unknown schedule status {other:?}")),
        }
    }
}

impl rusqlite::types::ToSql for ScheduleStatus {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl rusqlite::types::FromSql for ScheduleStatus {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        let s = value.as_str()?;
        match s {
            "active" => Ok(Self::Active),
            "cancelled" => Ok(Self::Cancelled),
            "completed" => Ok(Self::Completed),
            other => Err(rusqlite::types::FromSqlError::Other(
                format!("unknown schedule status {other:?}").into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Pending,
    Running,
    Completed,
    Failed,
}

impl RunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
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

impl std::str::FromStr for RunState {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(anyhow::anyhow!("unknown run state {other:?}")),
        }
    }
}

impl rusqlite::types::ToSql for RunState {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl rusqlite::types::FromSql for RunState {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        let s = value.as_str()?;
        match s {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(rusqlite::types::FromSqlError::Other(
                format!("unknown run state {other:?}").into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScheduleItemTiming {
    Once { at_ms: i64 },
    Recurring { rrule: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleItem {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub timing: ScheduleItemTiming,
    pub timezone: String,
    pub task_path: String,
    pub status: ScheduleStatus,
    pub next_run_at_ms: Option<i64>,
    pub created_at_ms: i64,
    pub cancelled_at_ms: Option<i64>,
}

impl ScheduleItem {
    pub fn schedule_type(&self) -> &'static str {
        match &self.timing {
            ScheduleItemTiming::Once { .. } => "once",
            ScheduleItemTiming::Recurring { .. } => "recurring",
        }
    }

    pub fn one_shot_at_ms(&self) -> Option<i64> {
        match &self.timing {
            ScheduleItemTiming::Once { at_ms } => Some(*at_ms),
            ScheduleItemTiming::Recurring { .. } => None,
        }
    }

    pub fn rrule(&self) -> Option<&str> {
        match &self.timing {
            ScheduleItemTiming::Once { .. } => None,
            ScheduleItemTiming::Recurring { rrule } => Some(rrule.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    let id: String = row.get(0)?;
    let schedule_id: String = row.get(1)?;
    let scheduled_for_ms: i64 = row.get(2)?;
    let started_at_ms: Option<i64> = row.get(3)?;
    let finished_at_ms: Option<i64> = row.get(4)?;
    let state_str: String = row.get(5)?;
    let state = match state_str.as_str() {
        "pending" => RunState::Pending,
        "running" => RunState::Running,
        "completed" => RunState::Completed,
        "failed" => RunState::Failed,
        other => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                5,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown run state {other:?}"),
                )),
            ));
        }
    };
    let codex_thread_id: Option<String> = row.get(6)?;
    let error: Option<String> = row.get(7)?;

    Ok(ScheduleRun {
        id,
        schedule_id,
        scheduled_for_ms,
        started_at_ms,
        finished_at_ms,
        state,
        codex_thread_id,
        error,
    })
}

fn schedule_from_row(row: &Row) -> rusqlite::Result<ScheduleItem> {
    let id: String = row.get(0)?;
    let name: String = row.get(1)?;
    let prompt: String = row.get(2)?;
    let schedule_type: String = row.get(3)?;
    let one_shot_at_ms: Option<i64> = row.get(4)?;
    let _dtstart_local: Option<String> = row.get(5)?;
    let rrule: Option<String> = row.get(6)?;
    let timezone: String = row.get(7)?;
    let task_path: String = row.get(8)?;
    let status_str: String = row.get(9)?;
    let status = match status_str.as_str() {
        "active" => ScheduleStatus::Active,
        "cancelled" => ScheduleStatus::Cancelled,
        "completed" => ScheduleStatus::Completed,
        other => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                9,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown schedule status {other:?}"),
                )),
            ));
        }
    };
    let next_run_at_ms: Option<i64> = row.get(10)?;
    let created_at_ms: i64 = row.get(11)?;
    let cancelled_at_ms: Option<i64> = row.get(12)?;

    let timing = match schedule_type.as_str() {
        "once" => {
            let at_ms = one_shot_at_ms.ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Null,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "one-shot schedule missing one_shot_at_ms",
                    )),
                )
            })?;
            ScheduleItemTiming::Once { at_ms }
        }
        "recurring" => {
            let rrule = rrule.ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    6,
                    rusqlite::types::Type::Null,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "recurring schedule missing rrule",
                    )),
                )
            })?;
            ScheduleItemTiming::Recurring { rrule }
        }
        other => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                3,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown schedule_type {other:?}"),
                )),
            ));
        }
    };

    Ok(ScheduleItem {
        id,
        name,
        prompt,
        timing,
        timezone,
        task_path,
        status,
        next_run_at_ms,
        created_at_ms,
        cancelled_at_ms,
    })
}

pub struct SchedulerDb;

impl SchedulerDb {
    /// Insert a schedule from timing that has already been validated.
    ///
    /// Takes the parsed [`ScheduleTiming`] rather than its five fields spread out:
    /// this used to be eleven parameters, most of them `&str` or `Option<i64>`, and
    /// two same-typed arguments in the wrong order is exactly the bug that once
    /// wrote `event_id = "whatsapp"` into every provider_ref row.
    pub fn create_schedule(
        runtime_db: &RuntimeDb,
        name: &str,
        prompt: &str,
        timing: &ScheduleTiming,
        task_path: &str,
    ) -> Result<ScheduleItem> {
        let id = format!("sched_{}", Uuid::new_v4().simple());
        let now_ms = Utc::now().timestamp_millis();

        let timing_spec = match timing {
            ScheduleTiming::Once { at_ms } => ScheduleItemTiming::Once { at_ms: *at_ms },
            ScheduleTiming::Recurring { rrule, .. } => ScheduleItemTiming::Recurring {
                rrule: rrule.clone(),
            },
        };

        let item = ScheduleItem {
            id: id.clone(),
            name: name.to_string(),
            prompt: prompt.to_string(),
            timing: timing_spec,
            // Cron is evaluated in the host's local time, so the only honest thing
            // to record is the offset that was in force when it was created. The
            // column used to hold an IANA name nothing ever read.
            timezone: Local::now().offset().to_string(),
            task_path: task_path.to_string(),
            status: ScheduleStatus::Active,
            next_run_at_ms: Some(timing.first_run_ms()),
            created_at_ms: now_ms,
            cancelled_at_ms: None,
        };

        // Save into SQLite
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO schedules (
                id, name, prompt, schedule_type, one_shot_at_ms, dtstart_local, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                item.id,
                item.name,
                item.prompt,
                item.schedule_type(),
                item.one_shot_at_ms(),
                None as Option<String>,
                item.rrule(),
                item.timezone,
                item.task_path,
                item.status.as_str(),
                item.next_run_at_ms,
                item.created_at_ms,
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
            params![Utc::now().timestamp_millis(), state.as_str(), error, run_id],
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

    pub fn update_next_run(
        runtime_db: &RuntimeDb,
        schedule_id: &str,
        next_run_at_ms: Option<i64>,
        status: Option<ScheduleStatus>,
    ) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        if let Some(st) = status {
            conn.execute(
                "UPDATE schedules SET next_run_at_ms = ?1, status = ?2 WHERE id = ?3",
                params![next_run_at_ms, st.as_str(), schedule_id],
            )?;
        } else {
            conn.execute(
                "UPDATE schedules SET next_run_at_ms = ?1 WHERE id = ?2",
                params![next_run_at_ms, schedule_id],
            )?;
        }
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
            "INSERT INTO schedules (id, name, prompt, schedule_type, timezone, task_path, status, created_at_ms)
             VALUES ('s1', 'name', 'prompt', 'recurring', 'UTC', 'path', 'bogus_status', 1000)",
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
