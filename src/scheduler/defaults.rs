//! The built-in schedules tera creates for itself.
//!
//! A fresh workspace gets a daily health pass and a nightly memory pass. Each
//! has its own durable marker, so cancelling one is not undone on restart and
//! adding a new one does not resurrect a cancelled old one.

use crate::runtime::RuntimeDb;
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::{self, ScheduleTiming};
use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use tracing::{info, warn};

/// Name, durable marker, cron rule, prompt, and where its working directory goes.
struct Builtin {
    name: &'static str,
    seeded_key: &'static str,
    rrule: &'static str,
    prompt: &'static str,
    task_dir: &'static str,
}

const BUILTINS: &[Builtin] = &[
    Builtin {
        name: "Machine health check",
        seeded_key: "seeded_self_care_schedule",
        rrule: "30 9 * * *",
        prompt: crate::data::SELF_CARE_PROMPT,
        task_dir: "tasks/machine-health",
    },
    Builtin {
        name: "Memory compaction",
        seeded_key: "seeded_memory_schedule",
        // Deep in the night, when a turn arriving mid pass is least likely.
        rrule: "0 3 * * *",
        prompt: crate::data::MEMORY_NIGHTLY_PROMPT,
        task_dir: "tasks/memory-compaction",
    },
];

/// Create the built-in schedules the first time this workspace starts.
///
/// Errors are logged, not propagated. A workspace that cannot seed housekeeping
/// still needs to come up and answer messages.
pub fn seed(runtime_db: &RuntimeDb) {
    for builtin in BUILTINS {
        if let Err(error) = try_seed(runtime_db, builtin) {
            warn!("Could not seed the {} schedule: {error:?}", builtin.name);
        }
    }
}

fn try_seed(runtime_db: &RuntimeDb, builtin: &Builtin) -> Result<()> {
    if runtime_db.get_state_value(builtin.seeded_key)?.is_some() {
        return Ok(());
    }

    let timing = ScheduleTiming::parse(
        &json!({ "type": "recurring", "rrule": builtin.rrule }),
        Utc::now().timestamp_millis(),
    )?;
    let item = SchedulerDb::create_schedule(
        runtime_db,
        builtin.name,
        builtin.prompt,
        &timing,
        builtin.task_dir,
    )?;
    runtime_db.set_state_value(builtin.seeded_key, &item.id)?;
    info!(
        target: "tera::scheduler",
        "Seeded the {} schedule ({}); first run {}",
        builtin.name,
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
    fn test_seeding_creates_every_builtin_schedule() {
        let runtime_db = db();
        seed(&runtime_db);

        let items = SchedulerDb::list_schedules(&runtime_db).unwrap();
        assert_eq!(items.len(), BUILTINS.len());
        for builtin in BUILTINS {
            let seeded = items
                .iter()
                .find(|item| item.name == builtin.name)
                .unwrap_or_else(|| panic!("{} was not seeded", builtin.name));
            assert_eq!(seeded.rrule.as_deref(), Some(builtin.rrule));
            assert!(seeded.next_run_at_ms.is_some(), "it would never fire");
        }

        let health = items
            .iter()
            .find(|item| item.name == "Machine health check")
            .unwrap();
        assert!(health.prompt.contains("SYSTEM.md"));
        let memory = items
            .iter()
            .find(|item| item.name == "Memory compaction")
            .unwrap();
        assert!(memory.prompt.contains("memory"));
    }

    #[test]
    fn test_seeding_twice_does_not_duplicate() {
        let runtime_db = db();
        seed(&runtime_db);
        seed(&runtime_db);
        assert_eq!(
            SchedulerDb::list_schedules(&runtime_db).unwrap().len(),
            BUILTINS.len()
        );
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
