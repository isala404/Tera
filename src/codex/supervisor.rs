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
use crate::history::db::HistoryDb;
use crate::runtime::{MainThreadState, RuntimeDb};
use anyhow::Result;
use chrono::Utc;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tracing::{error, info, warn};

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

    /// The live app-server, spawning and attaching the main thread if needed.
    ///
    /// CODEX_HOME points at the workspace so Codex loads the workspace
    /// `config.toml` plus Tera's process overrides (which register the MCP server) and the
    /// bootstrap `AGENTS.md`; cwd roots the thread in the workspace so
    /// `memories/`, `history/`, `projects/` and `tasks/` resolve.
    ///
    /// A manager whose process has exited is discarded and replaced rather than
    /// handed out again, the conversation outlives the process.
    pub async fn ensure(&self) -> Result<Arc<CodexProcessManager>> {
        let mut lock = self.mgr.lock().await;

        if let Some(mgr) = lock.as_ref() {
            if !mgr.is_dead() {
                return Ok(mgr.clone());
            }
            warn!("Codex app-server has exited; restarting it and reattaching the conversation");
            *lock = None;
        }

        let mgr = Arc::new(CodexProcessManager::spawn_for(&self.config).await?);

        let persisted = self.runtime_db.get_main_thread()?;
        let opts = ThreadOptions::new(&self.config.workspace_dir);
        let info = mgr
            .ensure_thread(persisted.as_ref().map(|s| s.thread_id.as_str()), &opts)
            .await?;

        let now_ms = Utc::now().timestamp_millis();
        let existing = persisted.as_ref().filter(|p| p.thread_id == info.id);
        let started_at_ms = existing.map(|p| p.started_at_ms).unwrap_or(now_ms);
        let last_activity_at_ms = existing.map(|p| p.last_activity_at_ms).unwrap_or(now_ms);
        let estimated_cache_warm_until_ms = existing
            .map(|p| p.estimated_cache_warm_until_ms)
            .unwrap_or(now_ms + crate::codex::CACHE_TTL_MS);

        self.runtime_db.save_main_thread(&MainThreadState {
            thread_id: info.id.clone(),
            turn_id: None,
            started_at_ms,
            last_activity_at_ms,
            estimated_cache_warm_until_ms,
            model_id: info.model.clone(),
        })?;

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
    /// Before starting, decide whether the existing thread is still the right
    /// place: a thread whose prompt cache has gone cold, or one started under a
    /// different model, is rotated out for a fresh one.
    pub async fn run_main_turn(&self, inputs: &[TurnInput]) -> Result<String> {
        let mgr = self.ensure().await?;
        let started_fresh = self.rotate_main_thread_if_stale(&mgr).await?;

        // A thread that starts empty does not know who it is talking to. It is
        // pointed at the workspace files rather than handed a summary of them
        //, then given the last few messages verbatim so the
        // rotation does not read as amnesia to the person on the other end.
        if started_fresh {
            let mut with_bootstrap = vec![TurnInput::Text(ThreadRouter::build_bootstrap_context(
                &self.config,
            ))];
            let recent = self.history_db.recent_messages(10)?;
            if !recent.is_empty() {
                with_bootstrap.push(TurnInput::Text(InputRenderer::render_history(&recent)));
            }
            with_bootstrap.extend_from_slice(inputs);
            return mgr.run_turn_inputs(&with_bootstrap).await;
        }

        mgr.run_turn_inputs(inputs).await
    }

    /// Apply the thread-selection policy to the main conversation.
    ///
    /// Returns whether the conversation is now on a thread with no prior context.
    async fn rotate_main_thread_if_stale(&self, mgr: &Arc<CodexProcessManager>) -> Result<bool> {
        let live_thread = mgr.active_thread().await;
        let model_id = self
            .runtime_db
            .get_main_thread()?
            .map(|state| state.model_id)
            .unwrap_or_default();

        match ThreadRouter::decide(&self.runtime_db, &model_id)? {
            ThreadDecision::Continue { thread_id } => {
                if live_thread.as_deref() != Some(thread_id.as_str()) {
                    // Persisted but not loaded in this process yet. If the resume
                    // fails, `ensure_thread` starts a new one, which needs the
                    // bootstrap, so report what actually happened.
                    let opts = ThreadOptions::new(&self.config.workspace_dir);
                    let info = mgr.ensure_thread(Some(&thread_id), &opts).await?;
                    return Ok(info.origin == ThreadOrigin::Created);
                }
                Ok(false)
            }
            ThreadDecision::Rotate { reason } => {
                info!("Rotating the main conversation onto a fresh thread: {reason}");
                let opts = ThreadOptions::new(&self.config.workspace_dir);
                let info = mgr.start_thread(&opts).await?;
                let now_ms = Utc::now().timestamp_millis();
                self.runtime_db.save_main_thread(&MainThreadState {
                    thread_id: info.id,
                    turn_id: None,
                    started_at_ms: now_ms,
                    last_activity_at_ms: now_ms,
                    estimated_cache_warm_until_ms: now_ms + crate::codex::CACHE_TTL_MS,
                    model_id: info.model,
                })?;
                Ok(true)
            }
        }
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

    /// A fresh thread rooted at `cwd`, not attached to the conversation.
    ///
    /// Returned as an id rather than run-and-forget so the caller can track
    /// and manage the isolated thread lifecycle.
    pub async fn start_isolated_thread(&self, cwd: &Path) -> Result<String> {
        self.start_isolated_thread_with_worker(cwd, None).await
    }

    pub async fn start_isolated_thread_with_worker(
        &self,
        cwd: &Path,
        worker_id: Option<&str>,
    ) -> Result<String> {
        let mgr = self.ensure().await?;
        let opts = match worker_id {
            Some(id) => ThreadOptions::with_worker(cwd, id),
            None => ThreadOptions::new(cwd),
        };
        let info = mgr.create_thread(&opts).await?;
        info!(
            "NEW isolated thread {} (model {}, worker {:?}) in {:?}. Separate from the conversation",
            info.id, info.model, worker_id, cwd
        );
        Ok(info.id)
    }

    pub async fn run_turn_on_thread(&self, thread_id: &str, prompt: &str) -> Result<String> {
        self.ensure()
            .await?
            .run_turn_on(thread_id, &[TurnInput::Text(prompt.to_string())])
            .await
    }

    /// Run a one-off turn on a fresh thread rooted at `cwd`.
    ///
    /// Returns the agent's final text, which for a scheduled task is a summary
    /// for the log, anything the user should see is sent by the agent itself
    /// through the `send_message` tool.
    pub async fn run_task_turn(&self, cwd: &Path, prompt: &str) -> Result<String> {
        self.run_task_turn_with_worker(cwd, prompt, None).await
    }

    pub async fn run_task_turn_with_worker(
        &self,
        cwd: &Path,
        prompt: &str,
        worker_id: Option<&str>,
    ) -> Result<String> {
        let thread_id = self
            .start_isolated_thread_with_worker(cwd, worker_id)
            .await?;
        let result = self.run_turn_on_thread(&thread_id, prompt).await;
        if let Err(error) = self.archive_thread(&thread_id).await {
            warn!("Could not archive isolated thread {thread_id}: {error:?}");
        }
        result
    }

    /// Ask the app-server which models it offers.
    pub async fn list_models(&self) -> Result<serde_json::Value> {
        self.ensure().await?.list_models().await
    }

    /// Interrupt whatever is running on a thread.
    pub async fn interrupt_thread(&self, thread_id: &str) -> Result<()> {
        let lock = self.mgr.lock().await;
        match lock.as_ref() {
            Some(mgr) if !mgr.is_dead() => mgr.interrupt(thread_id).await,
            _ => Ok(()),
        }
    }

    /// Release a completed isolated thread without touching the main
    /// conversation thread.
    pub async fn archive_thread(&self, thread_id: &str) -> Result<()> {
        let lock = self.mgr.lock().await;
        match lock.as_ref() {
            Some(mgr) if !mgr.is_dead() => mgr.archive_thread(thread_id).await,
            _ => Ok(()),
        }
    }
}
