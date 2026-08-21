use crate::codex::tier;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::runtime::{ActivityTracker, RuntimeDb};
use crate::scheduler::db::{ScheduleItem, ScheduleRun};
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::RecurrenceEngine;
use crate::workspace::templates;
use anyhow::Result;
use chrono::{Local, Utc};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

/// How often the scheduler looks for due work. Fine-grained enough for
/// minute-level schedules without spinning.
const TICK_INTERVAL: Duration = Duration::from_secs(5);

/// Past this much drift from its slot, a run is reported to its worker as late.
/// Comfortably above the tick interval so ordinary scheduling jitter is not
/// announced as an outage.
const LATE_THRESHOLD_MS: i64 = 60_000;

/// Upper bound on counting missed occurrences. A minutely rule after a week
/// offline has thousands, and the exact number stops mattering long before that.
const MAX_COUNTED_MISSES: usize = 100;

/// How far behind its schedule a run is.
struct Lateness {
    by_ms: i64,
    missed: usize,
}

pub struct SchedulerRunner {
    config: Config,
    runtime_db: RuntimeDb,
    codex: CodexSupervisor,
    activity: ActivityTracker,
    /// Schedules with a run in flight. A slow task must not be started again on
    /// the next tick five seconds later (PLAN.md section 28.1).
    running: Arc<Mutex<HashSet<String>>>,
}

impl SchedulerRunner {
    pub fn new(
        config: Config,
        runtime_db: RuntimeDb,
        codex: CodexSupervisor,
        activity: ActivityTracker,
    ) -> Self {
        Self {
            config,
            runtime_db,
            codex,
            activity,
            running: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn start_loop(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting background scheduler runner loop...");
            if let Err(e) = self.recover_stale_runs() {
                error!("Could not recover stale scheduled runs: {:?}", e);
            }
            loop {
                let now_ms = Utc::now().timestamp_millis();
                match SchedulerDb::get_due_schedules(&self.runtime_db, now_ms) {
                    Ok(due_items) => {
                        for item in due_items {
                            if !self.claim(&item.id) {
                                debug!("Schedule {} is still running; skipping this tick", item.id);
                                continue;
                            }

                            let self_clone = self.clone();
                            tokio::spawn(async move {
                                if let Err(e) = self_clone.run_schedule(&item).await {
                                    error!("Failed running schedule {}: {:?}", item.id, e);
                                }
                                self_clone.release(&item.id);
                            });
                        }
                    }
                    Err(e) => error!("Could not query due schedules: {:?}", e),
                }
                sleep(TICK_INTERVAL).await;
            }
        })
    }

    /// A process crash leaves schedule_runs in `running` after the schedule has
    /// already been advanced. Put that work back on the queue once and leave a
    /// note for the worker so it checks existing artifacts before repeating it.
    ///
    /// A run that cannot be recovered is logged and stepped over. One unwritable
    /// task directory must not strand every other stale run behind it.
    fn recover_stale_runs(&self) -> Result<()> {
        let now_ms = Utc::now().timestamp_millis();
        for run in SchedulerDb::running_runs(&self.runtime_db)? {
            if let Err(e) = self.recover_stale_run(&run, now_ms) {
                error!("Could not recover stale run {}: {:?}", run.id, e);
            }
        }
        Ok(())
    }

    fn recover_stale_run(&self, run: &ScheduleRun, now_ms: i64) -> Result<()> {
        let Some(item) = SchedulerDb::get_schedule(&self.runtime_db, &run.schedule_id)? else {
            return SchedulerDb::finish_run(
                &self.runtime_db,
                &run.id,
                "failed",
                Some("Phoenix found a run whose schedule no longer exists"),
            );
        };

        let task_dir = self.config.workspace_dir.join(&item.task_path);
        fs::create_dir_all(&task_dir)?;
        fs::write(
            task_dir.join("PHOENIX_RECOVERY.md"),
            format!(
                "Phoenix recovered schedule {} at {}. The previous process crashed while this run was marked running. Read MEMORY.md, RUNS.jsonl and artifacts before acting. Tell {} you recovered the run and continue only what remains.\n",
                item.name,
                Local::now().to_rfc3339(),
                self.config.owner_name,
            ),
        )?;

        SchedulerDb::finish_run(
            &self.runtime_db,
            &run.id,
            "failed",
            Some("Phoenix recovered this run after a daemon restart"),
        )?;

        if item.status == "active" || item.status == "completed" {
            SchedulerDb::update_next_run(&self.runtime_db, &item.id, Some(now_ms), Some("active"))?;
            warn!("Phoenix re-queued stale schedule '{}'", item.name);
        }

        Ok(())
    }

    /// Reserve a schedule for this tick; false if a run is already in flight.
    fn claim(&self, id: &str) -> bool {
        self.running.lock().unwrap().insert(id.to_string())
    }

    fn release(&self, id: &str) {
        self.running.lock().unwrap().remove(id);
    }

    pub async fn run_schedule(&self, item: &ScheduleItem) -> Result<()> {
        info!("Executing schedule {}: '{}'", item.id, item.name);

        // A scheduled run outranks memory maintenance, same as a conversation.
        let _active = self.activity.begin();

        let now_ms = Utc::now().timestamp_millis();

        // 1. Prepare schedule workspace
        let task_dir = self.config.workspace_dir.join(&item.task_path);
        fs::create_dir_all(&task_dir)?;

        let work_dir = task_dir.join("work");
        let artifacts_dir = task_dir.join("artifacts");
        fs::create_dir_all(&work_dir)?;
        fs::create_dir_all(&artifacts_dir)?;

        // 2. Ensure per-schedule MEMORY.md exists
        Self::migrate_legacy_task_files(&task_dir);
        let memory_md_path = task_dir.join("MEMORY.md");
        if !memory_md_path.exists() {
            fs::write(
                &memory_md_path,
                format!(
                    "# Schedule Memory: {}\n\nInitial schedule memory created at {}.\n",
                    item.name,
                    Local::now().to_rfc3339()
                ),
            )?;
        }

        let task_md_path = task_dir.join("TASK.md");
        fs::write(
            &task_md_path,
            format!("# Task Specification\n\nPrompt: {}\n", item.prompt),
        )?;

        // Codex reads AGENTS.md from the thread's cwd, so the bootstrap has to
        // live in the task directory itself (PLAN.md section 62).
        fs::write(
            task_dir.join("AGENTS.md"),
            templates::render(crate::data::SCHEDULE_AGENTS, &self.config),
        )?;

        // 3. Claim the run durably before advancing the schedule. A task that
        //    runs for minutes must not be re-fired by the ticks that happen
        //    while it works, and a crash in this small window must still leave
        //    Phoenix a running row to recover.
        //
        //    Computing the next run from *now* rather than from the missed slot is
        //    also what coalesces a backlog: after an outage a daily job fires once
        //    and resumes its normal cadence, instead of replaying 40 occurrences
        //    (PLAN.md section 28.2).
        let next_run = RecurrenceEngine::compute_next_run(
            &item.schedule_type,
            item.one_shot_at_ms,
            item.rrule.as_deref(),
            now_ms,
        )?;

        let lateness = Self::lateness(item, now_ms);
        if let Some(late) = &lateness {
            warn!(
                "Schedule {} ('{}') is running {} minutes late; {} occurrence(s) were missed",
                item.id,
                item.name,
                late.by_ms / 60_000,
                late.missed
            );
        }

        // 4. Actually run it, on a fresh Codex thread rooted in the task
        //    directory. Anything the user should see is sent by the agent itself
        //    via send_message; the returned text is a summary for the log.
        let run_id = SchedulerDb::start_run(
            &self.runtime_db,
            &item.id,
            item.next_run_at_ms.unwrap_or(now_ms),
        )?;

        let status = if next_run.is_none() {
            Some("completed")
        } else {
            None
        };
        SchedulerDb::update_next_run(&self.runtime_db, &item.id, next_run, status)?;

        let prompt = Self::build_task_prompt(&self.config.owner_name, item, &task_dir, lateness.as_ref());

        let tier = tier::by_name(&item.tier).unwrap_or_else(|e| {
            // A row with a tier this build does not know is a downgrade, not a
            // failure: the run still matters more than the model it runs on.
            warn!("Schedule {} has an unusable tier ({e}); running it routine", item.id);
            tier::ROUTINE
        });

        match self.codex.run_task_turn(&task_dir, &prompt, tier).await {
            Ok(summary) => {
                self.append_run_log(&task_dir, item, "completed", &summary);
                let _ = SchedulerDb::finish_run(&self.runtime_db, &run_id, "completed", None);
                let _ = fs::remove_file(task_dir.join("PHOENIX_RECOVERY.md"));
                info!(
                    "Schedule {} ('{}') completed. Next run: {:?}",
                    item.id, item.name, next_run
                );
            }
            Err(e) => {
                // Loud, and recorded in the task's own run log: a scheduled task
                // that silently fails is worse than one that never ran.
                self.append_run_log(&task_dir, item, "failed", &e.to_string());
                let _ = SchedulerDb::finish_run(
                    &self.runtime_db,
                    &run_id,
                    "failed",
                    Some(&e.to_string()),
                );
                let _ = fs::remove_file(task_dir.join("PHOENIX_RECOVERY.md"));
                error!("Schedule {} ('{}') failed: {:?}", item.id, item.name, e);
            }
        }

        Ok(())
    }

    /// How late this run is, and how many occurrences went by unrun.
    ///
    /// A run that fires within a tick or two of its slot is on time; anything
    /// beyond that means the daemon was down or busy, and the worker deserves to
    /// know before it reports "here is your morning brief" in the afternoon.
    fn lateness(item: &ScheduleItem, now_ms: i64) -> Option<Lateness> {
        let scheduled = item.next_run_at_ms?;
        let by_ms = now_ms - scheduled;
        if by_ms < LATE_THRESHOLD_MS {
            return None;
        }

        // Count the occurrences between the missed slot and now, walking the rule
        // forward. Bounded: after a week offline a minutely rule has thousands.
        let mut missed = 0usize;
        let mut cursor = scheduled;
        while missed < MAX_COUNTED_MISSES {
            match RecurrenceEngine::compute_next_run(
                &item.schedule_type,
                item.one_shot_at_ms,
                item.rrule.as_deref(),
                cursor,
            ) {
                Ok(Some(next)) if next <= now_ms && next > cursor => {
                    missed += 1;
                    cursor = next;
                }
                _ => break,
            }
        }

        Some(Lateness { by_ms, missed })
    }

    /// The prompt a scheduled worker wakes up to; text in
    /// `data/prompts/scheduled-task.md`.
    ///
    /// It has no conversation history, so it is told where it is, what it is for,
    /// and that reaching the user requires the `send_message` tool, returning
    /// text to nobody is the default failure mode otherwise.
    fn build_task_prompt(
        owner: &str,
        item: &ScheduleItem,
        task_dir: &Path,
        lateness: Option<&Lateness>,
    ) -> String {
        let late_note = match lateness {
            Some(late) => crate::data::render(
                crate::data::SCHEDULED_TASK_LATE_NOTE,
                &[
                    ("LATE_MINUTES", &(late.by_ms / 60_000).to_string()),
                    ("MISSED", &late.missed.to_string()),
                ],
            )
            .trim_end()
            .to_string(),
            None => String::new(),
        };

        crate::data::render(
            crate::data::SCHEDULED_TASK_PROMPT,
            &[
                ("OWNER", owner),
                ("TASK_NAME", &item.name),
                ("SCHEDULE_ID", &item.id),
                ("NOW", &Local::now().to_rfc3339()),
                ("TASK_DIR", &task_dir.display().to_string()),
                ("LATE_NOTE", &late_note),
                ("TASK_PROMPT", &item.prompt),
            ],
        )
    }

    /// Rename the pre-1.2 lowercase task files.
    ///
    /// Every knowledge file in the workspace is uppercase now, and a schedule that
    /// has been running for weeks has state in the old names. Renaming beats
    /// writing a fresh `MEMORY.md` beside a `memory.md` the worker will never read
    /// again, that silently loses everything previous runs recorded.
    ///
    /// macOS is case-insensitive, so `rename` there is a case correction and this
    /// is a no-op after the first pass. On Linux it is a real move.
    fn migrate_legacy_task_files(task_dir: &Path) {
        for (old, new) in [("memory.md", "MEMORY.md"), ("runs.jsonl", "RUNS.jsonl")] {
            let from = task_dir.join(old);
            let to = task_dir.join(new);
            if from.exists() && !to.exists() {
                if let Err(e) = fs::rename(&from, &to) {
                    warn!("Could not rename {:?} to {:?}: {e}", from, to);
                }
            }
        }
    }

    /// Append one line per run to the task's own `RUNS.jsonl` (PLAN.md 25).
    fn append_run_log(&self, task_dir: &Path, item: &ScheduleItem, state: &str, detail: &str) {
        let entry = serde_json::json!({
            "at": Local::now().to_rfc3339(),
            "schedule_id": item.id,
            "name": item.name,
            "state": state,
            "detail": detail.chars().take(2000).collect::<String>(),
        });

        let path = task_dir.join("RUNS.jsonl");
        let line = format!("{entry}\n");
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()))
        {
            warn!("Could not append to {:?}: {}", path, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_786_962_664_000;

    fn hourly(next_run_at_ms: Option<i64>) -> ScheduleItem {
        ScheduleItem {
            id: "sched_1".to_string(),
            name: "Hourly check".to_string(),
            prompt: "check".to_string(),
            schedule_type: "recurring".to_string(),
            one_shot_at_ms: None,
            dtstart_local: None,
            rrule: Some("EVERY_1H".to_string()),
            timezone: "UTC".to_string(),
            task_path: "tasks/schedule-1".to_string(),
            status: "active".to_string(),
            next_run_at_ms,
            created_at_ms: NOW,
            cancelled_at_ms: None,
            tier: tier::ROUTINE.name.to_string(),
        }
    }

    #[test]
    fn test_a_run_on_time_is_not_reported_as_late() {
        // Fired two ticks after its slot: ordinary jitter, not an outage.
        assert!(SchedulerRunner::lateness(&hourly(Some(NOW - 10_000)), NOW).is_none());
    }

    /// After an outage the missed occurrences collapse into one run, and the
    /// worker is told how many there were so it does not report stale work as
    /// current.
    #[test]
    fn test_an_outage_is_reported_with_the_occurrences_it_swallowed() {
        let five_hours_ago = NOW - 5 * 3600 * 1000;
        let late = SchedulerRunner::lateness(&hourly(Some(five_hours_ago)), NOW)
            .expect("five hours late");

        assert_eq!(late.by_ms / 60_000, 300);
        assert_eq!(late.missed, 5);
    }

    /// A long outage on a frequent rule must not spin counting occurrences.
    #[test]
    fn test_missed_counting_is_bounded() {
        let mut minutely = hourly(Some(NOW - 30 * 24 * 3600 * 1000));
        minutely.rrule = Some("EVERY_1M".to_string());

        let late = SchedulerRunner::lateness(&minutely, NOW).expect("very late");
        assert_eq!(late.missed, MAX_COUNTED_MISSES);
    }

    #[test]
    fn test_a_schedule_with_no_next_run_is_never_late() {
        assert!(SchedulerRunner::lateness(&hourly(None), NOW).is_none());
    }

    #[test]
    fn test_task_prompt_is_fully_rendered() {
        let item = hourly(Some(NOW));
        let prompt = SchedulerRunner::build_task_prompt(
            "Ada Lovelace",
            &item,
            Path::new("/ws/tasks/schedule-1"),
            None,
        );

        assert!(prompt.contains("Hourly check"));
        assert!(prompt.contains("sched_1"));
        assert!(prompt.contains("/ws/tasks/schedule-1"));
        assert!(prompt.contains("send_message"));
        assert!(!prompt.contains("LATE"));
        assert!(!prompt.contains("{{"), "unfilled placeholder: {prompt}");
    }

    /// A late run must say so in the prompt, with the numbers filled in, the
    /// worker cannot tell it is late any other way.
    #[test]
    fn test_a_late_run_is_told_how_late_it_is() {
        let item = hourly(Some(NOW));
        let late = Lateness {
            by_ms: 300 * 60_000,
            missed: 5,
        };
        let prompt = SchedulerRunner::build_task_prompt(
            "Ada Lovelace",
            &item,
            Path::new("/ws/tasks/schedule-1"),
            Some(&late),
        );

        assert!(prompt.contains("LATE by about 300 minutes"));
        assert!(prompt.contains("5 scheduled occurrence(s)"));
        assert!(!prompt.contains("{{"), "unfilled placeholder: {prompt}");
    }
}
