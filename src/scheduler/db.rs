use crate::codex::tier::{self, ModelTier};
use crate::runtime::{RuntimeDb, ScheduleItem, ScheduleRun};
use crate::scheduler::recurrence::ScheduleTiming;
use anyhow::Result;
use chrono::{Local, Utc};
use rusqlite::{params, Row};
use uuid::Uuid;

/// Both schedule queries select the same columns in the same order; sharing the
/// mapper is what keeps them from drifting when a column is added.
fn schedule_from_row(row: &Row) -> rusqlite::Result<ScheduleItem> {
    Ok(ScheduleItem {
        id: row.get(0)?,
        name: row.get(1)?,
        prompt: row.get(2)?,
        schedule_type: row.get(3)?,
        one_shot_at_ms: row.get(4)?,
        dtstart_local: row.get(5)?,
        rrule: row.get(6)?,
        timezone: row.get(7)?,
        task_path: row.get(8)?,
        status: row.get(9)?,
        next_run_at_ms: row.get(10)?,
        created_at_ms: row.get(11)?,
        cancelled_at_ms: row.get(12)?,
        // Rows written before tiers existed have NULL here. They were all created
        // when everything ran on one model, and the cheap tier is the safe default
        // for the recurring checks that make up most of them.
        tier: row
            .get::<_, Option<String>>(13)?
            .unwrap_or_else(|| tier::ROUTINE.name.to_string()),
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
        tier: ModelTier,
    ) -> Result<ScheduleItem> {
        let id = format!("sched_{}", Uuid::new_v4().simple());
        let now_ms = Utc::now().timestamp_millis();

        let item = ScheduleItem {
            id: id.clone(),
            name: name.to_string(),
            prompt: prompt.to_string(),
            schedule_type: timing.schedule_type.clone(),
            one_shot_at_ms: timing.one_shot_at_ms,
            dtstart_local: None,
            rrule: timing.rrule.clone(),
            // Cron is evaluated in the host's local time, so the only honest thing
            // to record is the offset that was in force when it was created. The
            // column used to hold an IANA name nothing ever read.
            timezone: Local::now().offset().to_string(),
            task_path: task_path.to_string(),
            status: "active".to_string(),
            next_run_at_ms: Some(timing.first_run_ms),
            created_at_ms: now_ms,
            cancelled_at_ms: None,
            tier: tier.name.to_string(),
        };

        // Save into SQLite
        let conn = runtime_db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO schedules (
                id, name, prompt, schedule_type, one_shot_at_ms, dtstart_local, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms, tier
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                item.id,
                item.name,
                item.prompt,
                item.schedule_type,
                item.one_shot_at_ms,
                item.dtstart_local,
                item.rrule,
                item.timezone,
                item.task_path,
                item.status,
                item.next_run_at_ms,
                item.created_at_ms,
                item.tier,
            ],
        )?;

        Ok(item)
    }

    pub fn list_schedules(runtime_db: &RuntimeDb) -> Result<Vec<ScheduleItem>> {
        let conn = runtime_db.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, prompt, schedule_type, one_shot_at_ms, dtstart_local, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms, cancelled_at_ms, tier
             FROM schedules WHERE status = 'active' ORDER BY created_at_ms ASC",
        )?;

        let rows = stmt.query_map([], schedule_from_row)?;

        let mut items = Vec::new();
        for r in rows {
            items.push(r?);
        }
        Ok(items)
    }

    /// Whether any schedule with this name has ever existed, in any state.
    ///
    /// Used to keep the seeded machine-health schedule from being recreated on
    /// every start after the owner cancels it.
    pub fn name_was_ever_used(runtime_db: &RuntimeDb, name: &str) -> Result<bool> {
        let conn = runtime_db.conn.lock().unwrap();
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM schedules WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )?;
        Ok(exists)
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
        let mut stmt = conn.prepare(
            "SELECT id, name, prompt, schedule_type, one_shot_at_ms, dtstart_local, rrule, timezone, task_path, status, next_run_at_ms, created_at_ms, cancelled_at_ms, tier
             FROM schedules WHERE status = 'active' AND next_run_at_ms IS NOT NULL AND next_run_at_ms <= ?1",
        )?;

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

    pub fn finish_run(
        runtime_db: &RuntimeDb,
        run_id: &str,
        state: &str,
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
        let mut stmt = conn.prepare(
            "SELECT id, schedule_id, scheduled_for_ms, started_at_ms, finished_at_ms, state, codex_thread_id, error
             FROM schedule_runs ORDER BY started_at_ms DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row: &Row| {
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
        })?;

        let mut runs = Vec::new();
        for r in rows {
            runs.push(r?);
        }
        Ok(runs)
    }

    pub fn update_next_run(
        runtime_db: &RuntimeDb,
        schedule_id: &str,
        next_run_at_ms: Option<i64>,
        status: Option<&str>,
    ) -> Result<()> {
        let conn = runtime_db.conn.lock().unwrap();
        if let Some(st) = status {
            conn.execute(
                "UPDATE schedules SET next_run_at_ms = ?1, status = ?2 WHERE id = ?3",
                params![next_run_at_ms, st, schedule_id],
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
