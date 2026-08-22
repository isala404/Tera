//! The built-in schedule tera creates for itself.
//!
//! A fresh workspace gets a daily health pass. Its durable marker means
//! cancelling it is not undone on restart.

use crate::codex::tier;
use crate::runtime::RuntimeDb;
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::{self, ScheduleTiming};
use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use tracing::{info, warn};

const SELF_CARE_NAME: &str = "Machine health check";
const SEEDED_KEY: &str = "seeded_self_care_schedule";
const SELF_CARE_RRULE: &str = "30 9 * * *";

/// Create the built-in schedules the first time this workspace starts.
///
/// Errors are logged, not propagated. A workspace that cannot seed housekeeping
/// still needs to come up and answer messages.
pub fn seed(runtime_db: &RuntimeDb) {
    if let Err(error) = try_seed(runtime_db) {
        warn!("Could not seed the {SELF_CARE_NAME} schedule: {error:?}");
    }
}

fn try_seed(runtime_db: &RuntimeDb) -> Result<()> {
    if runtime_db.get_state_value(SEEDED_KEY)?.is_some() {
        return Ok(());
    }

    // Adopt a schedule created by an older build rather than making a duplicate.
    if SchedulerDb::name_was_ever_used(runtime_db, SELF_CARE_NAME)? {
        runtime_db.set_state_value(SEEDED_KEY, "adopted")?;
        return Ok(());
    }

    let timing = ScheduleTiming::parse(
        &json!({ "type": "recurring", "rrule": SELF_CARE_RRULE }),
        Utc::now().timestamp_millis(),
    )?;
    let item = SchedulerDb::create_schedule(
        runtime_db,
        SELF_CARE_NAME,
        crate::data::SELF_CARE_PROMPT,
        &timing,
        "tasks/machine-health",
        tier::ROUTINE,
    )?;
    runtime_db.set_state_value(SEEDED_KEY, &item.id)?;
    info!(
        target: "tera::scheduler",
        "Seeded the {SELF_CARE_NAME} schedule ({}); first run {}",
        item.id,
        recurrence::local_time(timing.first_run_ms)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> RuntimeDb {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.sqlite3");
        std::mem::forget(dir);
        RuntimeDb::open(&path).unwrap()
    }

    #[test]
    fn test_seeding_creates_builtin_routine() {
        let runtime_db = db();
        seed(&runtime_db);

        let items = SchedulerDb::list_schedules(&runtime_db).unwrap();
        assert_eq!(items.len(), 1);
        let health = items
            .iter()
            .find(|item| item.name == SELF_CARE_NAME)
            .unwrap();
        assert_eq!(health.tier, tier::ROUTINE.name);
        assert_eq!(health.rrule.as_deref(), Some(SELF_CARE_RRULE));
        assert!(health.next_run_at_ms.is_some(), "it would never fire");
        assert!(health.prompt.contains("SYSTEM.md"));
    }

    #[test]
    fn test_seeding_twice_does_not_duplicate() {
        let runtime_db = db();
        seed(&runtime_db);
        seed(&runtime_db);
        assert_eq!(SchedulerDb::list_schedules(&runtime_db).unwrap().len(), 1);
    }

    #[test]
    fn test_cancelled_built_ins_are_not_recreated() {
        let runtime_db = db();
        seed(&runtime_db);
        for item in SchedulerDb::list_schedules(&runtime_db).unwrap() {
            assert!(SchedulerDb::cancel_schedule(&runtime_db, &item.id).unwrap());
        }

        seed(&runtime_db);
        assert!(SchedulerDb::list_schedules(&runtime_db).unwrap().is_empty());
    }
}
