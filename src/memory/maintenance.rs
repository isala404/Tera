//! Background maintenance: the nightly optimizer, and rebuilds triggered by a
//! model change.
//!
//! One loop owns both because they share the same precondition, nothing else is
//! using the app-server, and the same transaction shape. Its whole job is to
//! wait for an idle window and then not be in the way (PLAN.md sections 40, 41
//! and 11.2).

use crate::codex::models::{ModelDescriptor, ModelDiscovery};
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::history::db::HistoryDb;
use crate::memory::optimizer::{MemoryOptimizer, OptimizerOutcome, RETRY_PENDING_KEY};
use crate::memory::rebuild::{MemoryRebuilder, REBUILD_PENDING_KEY};
use crate::runtime::{ActivityTracker, RuntimeDb};
use anyhow::Result;
use chrono::{Local, Timelike};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

/// How often to look for an idle window. Maintenance is not urgent; polling
/// harder would only mean more chances to collide with real work.
const TICK: Duration = Duration::from_secs(60);

/// Local hours in which the nightly pass prefers to start: midnight to 04:00,
/// when a phone-based assistant is least likely to be mid-conversation. Written as
/// a range because `hour >= 0` is vacuously true, which clippy denies outright.
const NIGHTLY_WINDOW: std::ops::Range<u32> = 0..4;

/// After this long without a pass, the nightly window stops being a precondition.
/// A laptop that is shut every evening never sees 01:00, so preferring the window
/// unconditionally meant memory was optimized once, the first day, and never
/// again.
const STARVATION_DAYS: i64 = 3;

/// How often to re-check the default model (PLAN.md 11.2: startup, then daily).
const MODEL_CHECK_INTERVAL: Duration = Duration::from_secs(24 * 3600);

const LAST_OPTIMIZED_KEY: &str = "memory_optimizer_last_run_date";

pub struct MaintenanceRunner {
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    codex: CodexSupervisor,
    activity: ActivityTracker,
}

impl MaintenanceRunner {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        codex: CodexSupervisor,
        activity: ActivityTracker,
    ) -> Self {
        Self {
            config,
            history_db,
            runtime_db,
            codex,
            activity,
        }
    }

    pub fn start_loop(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting memory maintenance loop");

            // Check the model once at startup, before anything else, so a model
            // change detected here can be acted on in the same idle window.
            if let Err(e) = self.check_default_model().await {
                warn!("Could not read the model list at startup: {e:?}");
            }
            let mut last_model_check = tokio::time::Instant::now();

            loop {
                sleep(TICK).await;

                if last_model_check.elapsed() >= MODEL_CHECK_INTERVAL {
                    if let Err(e) = self.check_default_model().await {
                        warn!("Could not read the model list: {e:?}");
                    }
                    last_model_check = tokio::time::Instant::now();
                }

                if let Err(e) = self.tick().await {
                    error!("Memory maintenance tick failed: {e:?}");
                }
            }
        })
    }

    async fn tick(&self) -> Result<()> {
        // Nothing runs while the assistant is doing real work. Not "wait for a
        // gap and hope", the optimizer itself also aborts if work arrives after
        // it starts.
        if !self.activity.is_idle() {
            return Ok(());
        }

        if self.flag_is_set(REBUILD_PENDING_KEY)? {
            info!("Memory rebuild is pending; running it now that the system is idle");
            match MemoryRebuilder::run(
                &self.config,
                &self.history_db,
                &self.runtime_db,
                &self.codex,
            )
            .await
            {
                Ok(generation) => info!("Rebuild promoted generation {generation}"),
                // Leave the flag set: a failed rebuild should be retried, and
                // active memory is untouched either way.
                Err(e) => error!("Memory rebuild failed; active memory unchanged: {e:?}"),
            }
            return Ok(());
        }

        if !self.optimizer_is_due()? {
            return Ok(());
        }

        match MemoryOptimizer::run(
            &self.config,
            &self.runtime_db,
            &self.codex,
            &self.activity,
        )
        .await?
        {
            OptimizerOutcome::Promoted(generation) => {
                self.runtime_db
                    .set_state_value(LAST_OPTIMIZED_KEY, &today())?;
                info!("Nightly memory optimization promoted generation {generation}");
            }
            // Deliberately does not stamp the date: an abandoned pass should be
            // retried in the next idle window, not written off for the day.
            OptimizerOutcome::Interrupted => {
                info!("Memory optimization deferred; it will retry when idle")
            }
            OptimizerOutcome::Rejected(reason) => {
                warn!("Memory optimization rejected, active memory unchanged: {reason}")
            }
        }

        Ok(())
    }

    /// Due once per local day after the nightly hour, or immediately if a
    /// previous pass was abandoned.
    fn optimizer_is_due(&self) -> Result<bool> {
        if self.flag_is_set(RETRY_PENDING_KEY)? {
            return Ok(true);
        }

        let last_run = self.runtime_db.get_state_value(LAST_OPTIMIZED_KEY)?;
        if last_run.as_deref() == Some(today().as_str()) {
            return Ok(false);
        }

        Ok(Self::is_due(
            Local::now().hour(),
            last_run.as_deref(),
            Local::now().date_naive(),
        ))
    }

    /// Whether a pass may start now.
    ///
    /// A daemon started at noon should not immediately run maintenance for a day it
    /// was not up for, so the default is to wait for the nightly window. Two
    /// exceptions, both cases where waiting means never running: it has never run,
    /// and it has not run for days because the machine is never awake at 01:00.
    fn is_due(current_hour: u32, last_run: Option<&str>, today: chrono::NaiveDate) -> bool {
        let Some(last) = last_run else {
            return true;
        };
        if NIGHTLY_WINDOW.contains(&current_hour) {
            return true;
        }
        match last.parse::<chrono::NaiveDate>() {
            Ok(date) => (today - date).num_days() >= STARVATION_DAYS,
            // An unparseable date is a corrupt marker, not a reason to stop
            // maintaining memory forever.
            Err(_) => true,
        }
    }

    fn flag_is_set(&self, key: &str) -> Result<bool> {
        Ok(self.runtime_db.get_state_value(key)?.as_deref() == Some("true"))
    }

    /// Ask the app-server what it is running now, and flag a rebuild if the
    /// default model has changed since last time (PLAN.md section 11.2).
    async fn check_default_model(&self) -> Result<()> {
        let response = self.codex.list_models().await?;
        let models = Self::parse_models(&response);

        if models.is_empty() {
            warn!("model/list returned nothing recognisable: {}", response);
            return Ok(());
        }

        ModelDiscovery::process_models_response(&self.runtime_db, models)?;
        Ok(())
    }

    /// Pull descriptors out of a `model/list` response.
    ///
    /// Tolerant about shape on purpose: this is a Codex-side schema we do not own,
    /// and a field rename should degrade to "no model change detected" rather
    /// than take down the daemon.
    ///
    /// `data` is what codex-cli 0.147.0 actually returns; `models` and `items`
    /// are kept as fallbacks, and a bare array is accepted too.
    fn parse_models(response: &serde_json::Value) -> Vec<ModelDescriptor> {
        let items = response
            .get("data")
            .or_else(|| response.get("models"))
            .or_else(|| response.get("items"))
            .unwrap_or(response)
            .as_array();

        let Some(items) = items else {
            return Vec::new();
        };

        items
            .iter()
            .filter_map(|item| {
                let id = item
                    .get("id")
                    .or_else(|| item.get("model"))
                    .and_then(|i| i.as_str())?;
                Some(ModelDescriptor {
                    id: id.to_string(),
                    name: item
                        .get("displayName")
                        .or_else(|| item.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string),
                    is_default: item
                        .get("isDefault")
                        .or_else(|| item.get("is_default"))
                        .and_then(|d| d.as_bool())
                        .unwrap_or(false),
                })
            })
            .collect()
    }
}

fn today() -> String {
    Local::now().date_naive().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use serde_json::json;

    const TODAY: fn() -> NaiveDate = || NaiveDate::from_ymd_opt(2026, 8, 17).unwrap();

    #[test]
    fn test_never_optimized_runs_at_any_hour() {
        // Otherwise a machine that sleeps through every midnight never optimizes.
        assert!(MaintenanceRunner::is_due(14, None, TODAY()));
    }

    #[test]
    fn test_nightly_window_opens_after_midnight() {
        let yesterday = NaiveDate::from_ymd_opt(2026, 8, 16).unwrap().to_string();
        assert!(MaintenanceRunner::is_due(0, Some(&yesterday), TODAY()));
        assert!(MaintenanceRunner::is_due(3, Some(&yesterday), TODAY()));
        assert!(!MaintenanceRunner::is_due(14, Some(&yesterday), TODAY()));
        assert!(!MaintenanceRunner::is_due(4, Some(&yesterday), TODAY()));
    }

    /// The bug the window used to have: a laptop that is closed every night is
    /// never awake inside the window, so after the first pass memory was never
    /// optimized again.
    #[test]
    fn test_a_machine_never_awake_at_night_still_gets_a_pass() {
        let two_days = NaiveDate::from_ymd_opt(2026, 8, 15).unwrap().to_string();
        assert!(
            !MaintenanceRunner::is_due(14, Some(&two_days), TODAY()),
            "two days is still within the normal cadence"
        );

        let stale = NaiveDate::from_ymd_opt(2026, 8, 14).unwrap().to_string();
        assert!(MaintenanceRunner::is_due(14, Some(&stale), TODAY()));
    }

    #[test]
    fn test_a_corrupt_last_run_marker_does_not_stop_maintenance_forever() {
        assert!(MaintenanceRunner::is_due(14, Some("not-a-date"), TODAY()));
    }

    #[test]
    fn test_model_list_is_parsed() {
        let response = json!({
            "models": [
                {"id": "gpt-5.6-sol", "displayName": "GPT-5.6 Sol", "isDefault": true},
                {"id": "gpt-5.6-mini", "displayName": "GPT-5.6 mini"}
            ]
        });

        let models = MaintenanceRunner::parse_models(&response);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-5.6-sol");
        assert!(models[0].is_default);
        assert!(!models[1].is_default);
    }

    /// The shape codex-cli 0.147.0 really returns. It was parsed for `models`,
    /// found nothing, and logged the whole model list as unrecognisable on every
    /// startup, so a change of default model would never have been noticed.
    #[test]
    fn test_the_real_model_list_shape_is_parsed() {
        let response = json!({
            "data": [
                {"id": "gpt-5.6-sol", "model": "gpt-5.6-sol", "displayName": "GPT-5.6-Sol", "isDefault": true},
                {"id": "gpt-5.6-terra", "model": "gpt-5.6-terra", "displayName": "GPT-5.6-Terra", "isDefault": false}
            ],
            "nextCursor": null
        });

        let models = MaintenanceRunner::parse_models(&response);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-5.6-sol");
        assert!(models[0].is_default);
        assert!(!models[1].is_default);
    }

    /// A response shape we do not recognise must not panic or be mistaken for
    /// "the default model changed".
    #[test]
    fn test_unrecognised_model_list_yields_nothing() {
        assert!(MaintenanceRunner::parse_models(&json!({"unexpected": true})).is_empty());
        assert!(MaintenanceRunner::parse_models(&json!({"models": "gpt"})).is_empty());
        assert!(MaintenanceRunner::parse_models(&json!({"models": [{"noId": 1}]})).is_empty());
    }
}
