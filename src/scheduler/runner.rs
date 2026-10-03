use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::conversation::renderer::InputRenderer;
use crate::conversation::ConversationSession;
use crate::history::db::HistoryDb;
use crate::runtime::RuntimeDb;
use crate::scheduler::db::{RunState, ScheduleItem, ScheduleRun, ScheduleStatus, SchedulerDb};
use anyhow::Result;
use chrono::{Local, Utc};
use std::collections::{HashMap, HashSet};
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

/// How much of the chat a scheduled worker sees. Enough to notice that the
/// owner already said or did what the task is about to ask them.
const RECENT_CONVERSATION_MESSAGES: usize = 20;

/// Times one slot may be cut off by a crash before it is given up. A run that
/// takes the daemon down with it would otherwise crash it on every start.
const MAX_CRASHES_PER_SLOT: usize = 3;

/// How far behind its schedule a run is.
struct Lateness {
    by_ms: i64,
    missed: usize,
}

pub struct SchedulerRunner {
    config: Config,
    runtime_db: RuntimeDb,
    history_db: HistoryDb,
    codex: CodexSupervisor,
    session: ConversationSession,
    /// Schedules with a run in flight. A slow task must not be started again on
    /// the next tick five seconds later.
    running: Arc<Mutex<HashSet<String>>>,
}

impl SchedulerRunner {
    pub fn new(
        config: Config,
        runtime_db: RuntimeDb,
        history_db: HistoryDb,
        codex: CodexSupervisor,
        session: ConversationSession,
    ) -> Self {
        Self {
            config,
            runtime_db,
            history_db,
            codex,
            session,
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
    pub fn recover_stale_runs(&self) -> Result<()> {
        for run in SchedulerDb::running_runs(&self.runtime_db)? {
            if let Err(e) = self.recover_stale_run(&run) {
                error!("Could not recover stale run {}: {:?}", run.id, e);
            }
        }
        Ok(())
    }

    fn recover_stale_run(&self, run: &ScheduleRun) -> Result<()> {
        let Some(item) = SchedulerDb::get_schedule(&self.runtime_db, &run.schedule_id)? else {
            return SchedulerDb::finish_run(
                &self.runtime_db,
                &run.id,
                RunState::Failed,
                Some("Phoenix found a run whose schedule no longer exists"),
            );
        };

        let task_dir = self.config.workspace_dir.join(&item.task_path);
        fs::create_dir_all(&task_dir)?;

        // The run log is written before the database, so a crash between the
        // two leaves the log as the only record that the work finished.
        if self.run_log_contains(&task_dir, &run.id, "completed") {
            info!(
                "Run {} on schedule '{}' was completed prior to crash; reconciling database",
                run.id, item.name
            );
            SchedulerDb::finish_run(&self.runtime_db, &run.id, RunState::Completed, None)?;
            if item.next_run_at_ms.is_none() {
                SchedulerDb::complete_schedule(&self.runtime_db, &item.id)?;
            }
            let _ = fs::remove_file(task_dir.join("PHOENIX_RECOVERY.md"));
            return Ok(());
        }

        fs::write(
            task_dir.join("PHOENIX_RECOVERY.md"),
            format!(
                "Phoenix recovered schedule {} at {}. The previous process crashed while this run was marked running. Read MEMORY.md, RUNS.jsonl and artifacts before acting, and continue only what remains.\n",
                item.name,
                readable_now(),
            ),
        )?;

        SchedulerDb::finish_run(
            &self.runtime_db,
            &run.id,
            RunState::Failed,
            Some("Phoenix recovered this run after a daemon restart"),
        )?;

        if item.status != ScheduleStatus::Active {
            return Ok(());
        }
        let crashes =
            SchedulerDb::failed_runs_on_slot(&self.runtime_db, &item.id, run.scheduled_for_ms)?;
        if crashes < MAX_CRASHES_PER_SLOT {
            // Back on its original slot, so the retry is reported as late as it is.
            SchedulerDb::set_next_run(&self.runtime_db, &item.id, Some(run.scheduled_for_ms))?;
            warn!("Phoenix re-queued interrupted schedule '{}'", item.name);
            return Ok(());
        }
        let next = item.timing.next_run(Utc::now().timestamp_millis())?;
        SchedulerDb::set_next_run(&self.runtime_db, &item.id, next)?;
        if next.is_none() {
            SchedulerDb::complete_schedule(&self.runtime_db, &item.id)?;
        }
        error!(
            "Schedule '{}' crashed {crashes} times on one slot; skipped to {:?}",
            item.name, next
        );

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

        let now_ms = Utc::now().timestamp_millis();

        let task_dir = self.config.workspace_dir.join(&item.task_path);
        fs::create_dir_all(task_dir.join("work"))?;
        fs::create_dir_all(task_dir.join("artifacts"))?;

        let memory_md_path = task_dir.join("MEMORY.md");
        if !memory_md_path.exists() {
            fs::write(
                &memory_md_path,
                format!(
                    "# Schedule Memory: {}\n\nInitial schedule memory created at {}.\n",
                    item.name,
                    readable_now()
                ),
            )?;
        }
        fs::write(
            task_dir.join("TASK.md"),
            format!("# Task Specification\n\nPrompt: {}\n", item.prompt),
        )?;

        // Computing the next run from *now* rather than from the missed slot is
        // what coalesces a backlog: after an outage a daily job fires once and
        // resumes its normal cadence, instead of replaying 40 occurrences.
        let next_run = item.timing.next_run(now_ms)?;
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

        // Claim the run durably before advancing the schedule, so a crash in
        // between still leaves Phoenix a running row to recover, and the ticks
        // during a long run do not fire it again.
        let run_id = SchedulerDb::start_run(
            &self.runtime_db,
            &item.id,
            item.next_run_at_ms.unwrap_or(now_ms),
        )?;
        SchedulerDb::set_next_run(&self.runtime_db, &item.id, next_run)?;

        let prompt = self.build_task_prompt(item, &task_dir, lateness.as_ref())?;
        let worker_id = format!("schedule:{}", item.id);
        self.session.set_turn(&worker_id, Some(&run_id));
        let result = async {
            let thread_id = self.codex.start_task_thread(&task_dir, &worker_id).await?;
            SchedulerDb::set_run_codex_thread(&self.runtime_db, &run_id, &thread_id)?;
            info!(
                "Schedule {} ('{}') run {} started on thread {}",
                item.id, item.name, run_id, thread_id
            );
            self.codex.run_task_turn(&thread_id, &prompt).await
        }
        .await;
        self.session.set_turn(&worker_id, None);

        match &result {
            Ok(summary) => {
                self.append_run_log(&task_dir, item, &run_id, "completed", summary);
                SchedulerDb::finish_run(&self.runtime_db, &run_id, RunState::Completed, None)?;
                info!(
                    "Schedule {} ('{}') run {} completed. Next run: {:?}",
                    item.id, item.name, run_id, next_run
                );
            }
            Err(e) => {
                // Loud, and recorded in the task's own run log: a scheduled task
                // that silently fails is worse than one that never ran.
                self.append_run_log(&task_dir, item, &run_id, "failed", &e.to_string());
                SchedulerDb::finish_run(
                    &self.runtime_db,
                    &run_id,
                    RunState::Failed,
                    Some(&e.to_string()),
                )?;
                error!(
                    "Schedule {} ('{}') run {} failed: {:?}",
                    item.id, item.name, run_id, e
                );
            }
        }

        // A one-shot is spent whether its run worked or not; left active with
        // no next run it would sit in every schedule listing forever.
        if next_run.is_none() {
            SchedulerDb::complete_schedule(&self.runtime_db, &item.id)?;
        }
        let _ = fs::remove_file(task_dir.join("PHOENIX_RECOVERY.md"));
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
            match item.timing.next_run(cursor) {
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
    fn build_task_prompt(
        &self,
        item: &ScheduleItem,
        task_dir: &Path,
        lateness: Option<&Lateness>,
    ) -> Result<String> {
        let recent = self
            .history_db
            .recent_messages(RECENT_CONVERSATION_MESSAGES)?;
        let conversation = if recent.is_empty() {
            "Nothing yet.".to_string()
        } else {
            InputRenderer::render_history("", &recent, &HashMap::new())
        };
        Ok(render_task_prompt(
            &self.config,
            item,
            task_dir,
            lateness,
            &conversation,
        ))
    }

    /// Append one line per run to the task's own `RUNS.jsonl`.
    fn append_run_log(
        &self,
        task_dir: &Path,
        item: &ScheduleItem,
        run_id: &str,
        state: &str,
        detail: &str,
    ) {
        let entry = serde_json::json!({
            "at": Local::now().to_rfc3339(),
            "run_id": run_id,
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

    fn run_log_contains(&self, task_dir: &Path, run_id: &str, state: &str) -> bool {
        let path = task_dir.join("RUNS.jsonl");
        if let Ok(content) = fs::read_to_string(&path) {
            for line in content.lines() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    if val.get("run_id").and_then(|v| v.as_str()) == Some(run_id)
                        && val.get("state").and_then(|v| v.as_str()) == Some(state)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

fn render_task_prompt(
    config: &Config,
    item: &ScheduleItem,
    task_dir: &Path,
    lateness: Option<&Lateness>,
    conversation: &str,
) -> String {
    let timing = match lateness {
        Some(late) => format!(
            "late by about {} minutes, {} occurrence(s) missed and coalesced into this run",
            late.by_ms / 60_000,
            late.missed
        ),
        None => "on time".to_string(),
    };

    crate::data::render(
        crate::data::SCHEDULED_TASK_PROMPT,
        &[
            ("OWNER", &config.owner_name),
            ("WORKSPACE", &config.workspace_dir.display().to_string()),
            ("TASK_NAME", &item.name),
            ("SCHEDULE_ID", &item.id),
            ("NOW", &readable_now()),
            ("TASK_DIR", &task_dir.display().to_string()),
            ("LATE", &timing),
            ("TASK_PROMPT", &item.prompt),
            ("RECENT_CONVERSATION", conversation),
        ],
    )
}

/// The weekday is what a model gets wrong from a bare ISO timestamp.
fn readable_now() -> String {
    Local::now().format("%A %Y-%m-%d %H:%M %:z").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ScheduleTiming;

    const NOW: i64 = 1_786_962_664_000;

    fn owner_config() -> Config {
        let mut config = Config::new(std::path::PathBuf::from("/ws"), true);
        config.owner_name = "Ada Lovelace".to_string();
        config
    }

    fn hourly(next_run_at_ms: Option<i64>) -> ScheduleItem {
        ScheduleItem {
            id: "sched_1".to_string(),
            name: "Hourly check".to_string(),
            prompt: "check".to_string(),
            timing: ScheduleTiming::Recurring {
                rrule: "EVERY_1H".to_string(),
            },
            task_path: "tasks/schedule-1".to_string(),
            status: ScheduleStatus::Active,
            next_run_at_ms,
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
        let late =
            SchedulerRunner::lateness(&hourly(Some(five_hours_ago)), NOW).expect("five hours late");

        assert_eq!(late.by_ms / 60_000, 300);
        assert_eq!(late.missed, 5);
    }

    /// A long outage on a frequent rule must not spin counting occurrences.
    #[test]
    fn test_missed_counting_is_bounded() {
        let mut minutely = hourly(Some(NOW - 30 * 24 * 3600 * 1000));
        minutely.timing = ScheduleTiming::Recurring {
            rrule: "EVERY_1M".to_string(),
        };

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
        let prompt = render_task_prompt(
            &owner_config(),
            &item,
            Path::new("/ws/tasks/schedule-1"),
            None,
            "[2026-08-18 09:00:00 +05:30] User m_1:\nsaid something",
        );

        assert!(prompt.contains("Hourly check"));
        assert!(prompt.contains("sched_1"));
        assert!(prompt.contains("/ws/tasks/schedule-1"));
        assert!(prompt.contains("send_message"));
        assert!(prompt.contains("on time"));
        assert!(prompt.contains("said something"));
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
        let prompt = render_task_prompt(
            &owner_config(),
            &item,
            Path::new("/ws/tasks/schedule-1"),
            Some(&late),
            "Nothing yet.",
        );

        assert!(prompt.contains("late by about 300 minutes"));
        assert!(prompt.contains("5 occurrence(s) missed"));
        assert!(!prompt.contains("{{"), "unfilled placeholder: {prompt}");
    }
}
