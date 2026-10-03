//! Single owner of the `codex app-server` child process.
//!
//! Both the turn engine (foreground conversation) and the scheduler (background
//! tasks) need to run Codex turns. Running two app-server processes would double
//! the memory and split the model's view of the workspace, so one supervisor owns
//! the process and hands out turns on it:
//!
//! * the main conversation lives on one long-lived thread, persisted and resumed
//!   across restarts;
//! * each scheduled run gets a fresh thread rooted in its own task directory, so
//!   background work never pollutes the conversation the user is having.

use crate::codex::process::{ThreadOptions, ThreadOrigin, TurnInput};
use crate::codex::thread_router::{ThreadDecision, ThreadRouter};
use crate::codex::CodexProcessManager;
use crate::config::Config;
use crate::conversation::renderer::InputRenderer;
use crate::history::db::{ConversationEvent, HistoryDb};
use crate::runtime::{MainThreadState, RuntimeDb};
use crate::scheduler::db::SchedulerDb;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tracing::{error, info, warn};

/// How much of the conversation a fresh thread is handed, and the most a warm
/// one is shown of what it missed.
const RECENT_HISTORY_MESSAGES: usize = 30;

/// The newest history `seq` the main thread has been given, so the next turn
/// can show it what arrived from elsewhere after that.
const MAIN_SEEN_THROUGH_KEY: &str = "main_thread_seen_through_seq";

#[derive(Clone)]
pub struct CodexSupervisor {
    config: Config,
    runtime_db: RuntimeDb,
    history_db: HistoryDb,
    mgr: Arc<Mutex<Option<Arc<CodexProcessManager>>>>,
    active_login: Arc<Mutex<Option<(String, String)>>>,
}

impl CodexSupervisor {
    pub fn new(config: Config, runtime_db: RuntimeDb, history_db: HistoryDb) -> Self {
        Self {
            config,
            runtime_db,
            history_db,
            mgr: Arc::new(Mutex::new(None)),
            active_login: Arc::new(Mutex::new(None)),
        }
    }

    /// Start connecting now so the first message does not pay the spawn cost.
    /// Failure is not fatal. `ensure` retries on demand.
    pub fn warm_in_background(&self) {
        let this = self.clone();
        tokio::spawn(async move {
            info!("Bootstrapping persistent Codex app-server process on daemon startup...");
            match this.ensure().await {
                Ok(_) => info!("Codex app-server process ready and connected!"),
                Err(e) => error!("Failed to bootstrap Codex app-server on startup: {:?}", e),
            }
        });
    }

    /// Check whether Codex currently has valid authentication.
    ///
    /// Not gated on `cfg(test)`: integration tests link this crate compiled
    /// without it, so a test that drove a turn to completion would spawn a real
    /// app-server against the operator's own account.
    pub async fn check_authenticated(&self) -> bool {
        if self.config.mock_transport {
            return true;
        }
        match self.ensure().await {
            Ok(mgr) => mgr.is_authenticated().await,
            Err(_) => false,
        }
    }

    /// Request an OAuth device code for pairing.
    /// Returns `(verification_url, user_code)`. Caches the active request until completed or cleared.
    pub async fn request_device_login(&self) -> Result<(String, String)> {
        let mut lock = self.active_login.lock().await;
        if let Some(cached) = lock.as_ref() {
            return Ok(cached.clone());
        }

        let mgr = self.ensure().await?;
        let (url, code) = mgr.start_device_login().await?;
        *lock = Some((url.clone(), code.clone()));
        Ok((url, code))
    }

    /// Clear the cached device login code (e.g. after login finishes).
    pub async fn clear_active_login(&self) {
        let mut lock = self.active_login.lock().await;
        *lock = None;
    }

    /// Subscribe to login completion notifications from the app-server.
    pub async fn subscribe_login_completed(&self) -> Result<broadcast::Receiver<bool>> {
        let mgr = self.ensure().await?;
        Ok(mgr.subscribe_login_completed())
    }

    /// The live app-server, spawning it if needed.
    ///
    /// CODEX_HOME points at the workspace so Codex loads the workspace
    /// `config.toml` plus Tera's process overrides (which register the MCP
    /// server) and the bootstrap `AGENTS.md`.
    ///
    /// A manager whose process has exited is discarded and replaced rather than
    /// handed out again. The main thread is attached by the next main turn.
    pub async fn ensure(&self) -> Result<Arc<CodexProcessManager>> {
        let mut lock = self.mgr.lock().await;

        if let Some(mgr) = lock.as_ref() {
            if !mgr.is_dead() {
                return Ok(mgr.clone());
            }
            warn!("Codex app-server has exited; restarting it");
            *lock = None;
        }

        let mgr = Arc::new(CodexProcessManager::spawn_for(&self.config).await?);
        *lock = Some(mgr.clone());
        Ok(mgr)
    }

    /// Whether a turn is running on the main conversation thread right now.
    pub async fn main_turn_is_running(&self) -> bool {
        let lock = self.mgr.lock().await;
        let Some(mgr) = lock.as_ref() else {
            return false;
        };
        if mgr.is_dead() {
            return false;
        }
        let Some(thread_id) = mgr.active_thread().await else {
            return false;
        };
        mgr.active_turn_of(&thread_id).await.is_some()
    }

    /// Feed input into the turn already running on the main thread.
    ///
    /// `Ok(false)` means there was nothing to steer, the turn finished in the
    /// gap between the check and the call, and the caller should start a new
    /// turn instead. No input may be lost either way.
    pub async fn steer_main_turn(&self, inputs: &[TurnInput]) -> Result<bool> {
        let mgr = {
            let lock = self.mgr.lock().await;
            match lock.as_ref() {
                Some(mgr) if !mgr.is_dead() => mgr.clone(),
                _ => return Ok(false),
            }
        };

        let Some(thread_id) = mgr.active_thread().await else {
            return Ok(false);
        };

        match mgr.steer(&thread_id, inputs).await {
            Ok(()) => Ok(true),
            Err(e) => {
                info!("Could not steer the running turn ({e}); it will start a new one");
                Ok(false)
            }
        }
    }

    /// Run a turn on the main conversation thread.
    ///
    /// A thread that starts empty does not know who it is talking to. It is
    /// pointed at the workspace files rather than handed a summary of them, then
    /// given the recent conversation verbatim so the rotation does not read as
    /// amnesia to the person on the other end. A thread that carries on is
    /// shown what was said in the chat from outside it since its last turn,
    /// which is how it learns what a scheduled task told the owner.
    pub async fn run_main_turn(&self, inputs: &[TurnInput]) -> Result<String> {
        let mgr = self.ensure().await?;
        let started_fresh = self.attach_main_thread(&mgr).await?;
        let seen_through = self.history_db.latest_seq()?;

        let context = if started_fresh {
            let mut context = vec![TurnInput::Text(ThreadRouter::build_bootstrap_context(
                &self.config,
            ))];
            let recent = self.history_db.recent_messages(RECENT_HISTORY_MESSAGES)?;
            if !recent.is_empty() {
                context.push(TurnInput::Text(InputRenderer::render_history(
                    "Recent conversation from history",
                    &recent,
                    &self.outside_sources(&recent)?,
                )));
            }
            context
        } else {
            self.unseen_messages()?
                .map(TurnInput::Text)
                .into_iter()
                .collect()
        };

        self.runtime_db
            .set_state_value(MAIN_SEEN_THROUGH_KEY, &seen_through.to_string())?;
        let with_context: Vec<TurnInput> =
            context.into_iter().chain(inputs.iter().cloned()).collect();
        mgr.run_turn_inputs(&with_context).await
    }

    /// Messages sent into the chat from outside the main thread since its last
    /// turn, rendered for it, or `None` when there are none.
    fn unseen_messages(&self) -> Result<Option<String>> {
        let Some(seen_through) = self
            .runtime_db
            .get_state_value(MAIN_SEEN_THROUGH_KEY)?
            .and_then(|value| value.parse::<i64>().ok())
        else {
            return Ok(None);
        };
        let recent = self
            .history_db
            .assistant_messages_after(seen_through, RECENT_HISTORY_MESSAGES)?;
        let sources = self.outside_sources(&recent)?;
        let unseen: Vec<ConversationEvent> = recent
            .into_iter()
            .filter(|event| sources.contains_key(&event.id))
            .collect();
        if unseen.is_empty() {
            return Ok(None);
        }
        Ok(Some(InputRenderer::render_history(
            "Sent in this chat since your last turn, from outside this thread",
            &unseen,
            &sources,
        )))
    }

    /// Where each assistant message that did not come from the main thread was
    /// sent from, keyed by event id. Messages the main thread sent itself, and
    /// every user message, are absent.
    fn outside_sources(&self, events: &[ConversationEvent]) -> Result<HashMap<String, String>> {
        let mut sources = HashMap::new();
        for event in events.iter().filter(|event| event.actor == "assistant") {
            let source = match event.turn_id() {
                Some(turn_id) if self.runtime_db.get_turn(turn_id)?.is_some() => continue,
                Some(turn_id) => SchedulerDb::get_run(&self.runtime_db, turn_id)?
                    .map(|run| SchedulerDb::get_schedule(&self.runtime_db, &run.schedule_id))
                    .transpose()?
                    .flatten()
                    .map(|schedule| {
                        format!("scheduled task \"{}\" ({})", schedule.name, schedule.id)
                    }),
                None => None,
            };
            sources.insert(
                event.id.clone(),
                source.unwrap_or_else(|| "outside this thread".to_string()),
            );
        }
        Ok(sources)
    }

    /// Put the right thread under the main conversation. Returns whether it
    /// has no prior context.
    async fn attach_main_thread(&self, mgr: &Arc<CodexProcessManager>) -> Result<bool> {
        let persisted = self.runtime_db.get_main_thread()?;
        let now_ms = Utc::now().timestamp_millis();
        let opts = ThreadOptions::new(&self.config.workspace_dir);

        let info = match ThreadRouter::decide(persisted.as_ref(), now_ms) {
            ThreadDecision::Continue { thread_id } => {
                if mgr.active_thread().await.as_deref() == Some(thread_id.as_str()) {
                    return Ok(false);
                }
                // Persisted but not loaded in this process yet. If the resume
                // fails, `ensure_thread` starts a new one.
                let info = mgr.ensure_thread(Some(&thread_id), &opts).await?;
                if info.origin == ThreadOrigin::Resumed {
                    return Ok(false);
                }
                info
            }
            ThreadDecision::Rotate { reason } => {
                info!("Starting a fresh main conversation thread: {reason}");
                mgr.start_thread(&opts).await?
            }
        };

        self.runtime_db.save_main_thread(&MainThreadState {
            thread_id: info.id,
            started_at_ms: now_ms,
            last_activity_at_ms: now_ms,
            estimated_cache_warm_until_ms: now_ms + crate::codex::CACHE_TTL_MS,
            model_id: info.model,
        })?;
        Ok(true)
    }

    /// Record activity so the cache-warm estimate slides forward.
    pub fn note_main_activity(&self) {
        if let Ok(Some(mut state)) = self.runtime_db.get_main_thread() {
            let now_ms = Utc::now().timestamp_millis();
            state.last_activity_at_ms = now_ms;
            state.estimated_cache_warm_until_ms = now_ms + crate::codex::CACHE_TTL_MS;
            if let Err(e) = self.runtime_db.save_main_thread(&state) {
                warn!("Could not update main thread activity: {e}");
            }
        }
    }

    /// A fresh thread rooted at `cwd`, separate from the conversation. Tool
    /// calls made on it are attributed to `worker_id`.
    pub async fn start_task_thread(&self, cwd: &Path, worker_id: &str) -> Result<String> {
        let info = self
            .ensure()
            .await?
            .create_thread(&ThreadOptions::with_worker(cwd, worker_id))
            .await?;
        info!(
            "NEW isolated thread {} (model {}, worker {worker_id}) in {:?}",
            info.id, info.model, cwd
        );
        Ok(info.id)
    }

    /// Run the one turn a task thread exists for, then archive it.
    ///
    /// Returns the agent's final text, which for a task is a summary for the
    /// log. Anything the user should see is sent by the agent itself through
    /// the `send_message` tool.
    pub async fn run_task_turn(&self, thread_id: &str, prompt: &str) -> Result<String> {
        let mgr = self.ensure().await?;
        let result = mgr
            .run_turn_on(thread_id, &[TurnInput::Text(prompt.to_string())])
            .await;
        if let Err(error) = mgr.archive_thread(thread_id).await {
            warn!("Could not archive isolated thread {thread_id}: {error:?}");
        }
        result
    }
}
