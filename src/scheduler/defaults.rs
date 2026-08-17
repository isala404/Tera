//! The one schedule tera creates for itself.
//!
//! The assistant is told to keep the machine it lives on healthy, but it only
//! wakes when a message arrives, so without a seeded schedule the instruction is
//! aspiration, not behaviour. A fresh workspace gets a daily health pass and looks
//! after the host from the first start.
//!
//! Seeding is recorded in `daemon_state` rather than inferred from the schedules
//! table. Cancelling it has to stick: keying off "is there an active schedule with
//! this name" would recreate it on the very next restart, which is the assistant
//! overruling the user once a day forever.

use crate::codex::tier;
use crate::runtime::RuntimeDb;
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::{self, ScheduleTiming};
use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use tracing::{info, warn};

const SELF_CARE_NAME: &str = "Machine health check";

/// Marks that seeding has happened. Survives cancellation of the schedule itself.
const SEEDED_KEY: &str = "seeded_self_care_schedule";

/// 09:30 local, daily. A laptop is usually asleep at 4am, and a pass that is
/// always late is a pass whose lateness stops meaning anything; mid-morning it
/// mostly runs on time, and the prompt tells it that being late does not matter
/// for this particular task anyway.
const SELF_CARE_RRULE: &str = "30 9 * * *";

/// Create the machine-health schedule the first time this workspace starts.
///
/// Errors are logged, not propagated: a workspace that cannot seed a housekeeping
/// task is still a workspace that should come up and answer messages.
pub fn seed(runtime_db: &RuntimeDb) {
    if let Err(e) = try_seed(runtime_db) {
        warn!("Could not seed the {SELF_CARE_NAME} schedule: {e:?}");
    }
}

fn try_seed(runtime_db: &RuntimeDb) -> Result<()> {
    if runtime_db.get_state_value(SEEDED_KEY)?.is_some() {
        return Ok(());
    }

    // A workspace that predates this code already has the schedule under this name
    // if a previous build seeded it by name. Adopt it rather than making a second.
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
        // The tempdir has to outlive the connection; leaking it is fine in a test.
        std::mem::forget(dir);
        RuntimeDb::open(&path).unwrap()
    }

    #[test]
    fn test_seeding_creates_one_daily_routine_schedule() {
        let runtime_db = db();
        seed(&runtime_db);

        let items = SchedulerDb::list_schedules(&runtime_db).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, SELF_CARE_NAME);
        assert_eq!(items[0].tier, tier::ROUTINE.name);
        assert_eq!(items[0].rrule.as_deref(), Some(SELF_CARE_RRULE));
        assert!(items[0].next_run_at_ms.is_some(), "it would never fire");
        assert!(items[0].prompt.contains("SYSTEM.md"));
    }

    #[test]
    fn test_seeding_twice_does_not_make_two() {
        let runtime_db = db();
        seed(&runtime_db);
        seed(&runtime_db);
        assert_eq!(SchedulerDb::list_schedules(&runtime_db).unwrap().len(), 1);
    }

    /// The important one. Every start calls `seed`, so a cancellation that does not
    /// stick means the assistant reinstates a task the user deliberately removed.
    #[test]
    fn test_a_cancelled_seed_is_not_recreated_on_the_next_start() {
        let runtime_db = db();
        seed(&runtime_db);

        let id = SchedulerDb::list_schedules(&runtime_db).unwrap()[0].id.clone();
        assert!(SchedulerDb::cancel_schedule(&runtime_db, &id).unwrap());

        seed(&runtime_db);
        assert!(
            SchedulerDb::list_schedules(&runtime_db).unwrap().is_empty(),
            "the health schedule came back after being cancelled"
        );
    }
}
