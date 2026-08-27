//! Model-led startup reporting and bounded recovery.
//!
//! Rust supplies facts and enforces retry limits. The startup prompt decides how
//! to explain a restart, what needs checking, and whether a follow-up is useful.

use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::conversation::record_assistant_message;
use crate::conversation::renderer::InputRenderer;
use crate::conversation::session::ConversationSession;
use crate::data;
use crate::history::db::HistoryDb;
use crate::runtime::crash_mark::CrashMark;
use crate::runtime::{ConversationTurn, RuntimeDb};
use crate::transport::{MessageRef, Transport};
use crate::update::UpdateNotice;
use crate::version::BuildInfo;
use anyhow::{bail, Result};
use serde_json::json;
use std::sync::Arc;
use tracing::{info, warn};

const MAX_TURN_ATTEMPTS: i64 = 2;
const MAX_CONSECUTIVE_CRASHES: u32 = 2;

pub struct Phoenix {
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    transport: Arc<dyn Transport>,
    codex: CodexSupervisor,
    session: ConversationSession,
}

impl Phoenix {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        transport: Arc<dyn Transport>,
        codex: CodexSupervisor,
        session: ConversationSession,
    ) -> Self {
        Self {
            config,
            history_db,
            runtime_db,
            transport,
            codex,
            session,
        }
    }

    /// Run once after every start. With no prior chat there is nowhere to speak,
    /// which is normal on a brand-new workspace.
    pub async fn run(
        &self,
        crashed: Option<CrashMark>,
        update: Option<UpdateNotice>,
    ) -> Result<()> {
        let pending = self.runtime_db.unfinished_turns()?;
        let Some(chat_jid) = self.chat_to_speak_into(&pending)? else {
            info!("Startup assistant has no previous conversation yet");
            return Ok(());
        };

        // Everything below needs a model turn. The crash mark and the update
        // journal are both consumed by this start, so returning early here would
        // lose the report for good rather than defer it: pair first, then run.
        if !self.codex.check_authenticated().await {
            self.pair_with_codex(&chat_jid).await?;
        }

        let over_budget = crashed
            .as_ref()
            .is_some_and(|mark| mark.consecutive >= MAX_CONSECUTIVE_CRASHES);
        let (recoverable, abandoned): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|turn| !over_budget && turn.attempts < MAX_TURN_ATTEMPTS);

        self.recover(
            &chat_jid,
            crashed.as_ref(),
            update.as_ref(),
            &recoverable,
            &abandoned,
            over_budget,
        )
        .await?;

        for turn in recoverable.iter().chain(&abandoned) {
            self.runtime_db.record_turn_attempt(&turn.turn_id)?;
        }
        for turn in &recoverable {
            self.runtime_db.finish_turn(&turn.turn_id, "completed")?;
        }
        for turn in &abandoned {
            self.runtime_db.finish_turn(&turn.turn_id, "abandoned")?;
        }

        let restart_context = self.config.runtime_dir().join("restart-context.md");
        if let Err(error) = std::fs::remove_file(&restart_context) {
            if error.kind() != std::io::ErrorKind::NotFound {
                warn!("Could not clear startup context: {error}");
            }
        }
        Ok(())
    }

    /// Walk the owner through device pairing, and wait for it to resolve.
    ///
    /// Waiting rather than returning is what keeps the restart report and the
    /// interrupted turns intact: both are answered by a model turn, and both
    /// pieces of evidence are gone by the next start. Phoenix already runs on its
    /// own task, so the wait costs nothing else, and it is bounded by Codex,
    /// which reports the fifteen-minute expiry as a failed login. Failing puts
    /// this back in the caller's retry loop, which starts over with a fresh code.
    async fn pair_with_codex(&self, chat_jid: &str) -> Result<()> {
        info!("Codex is unauthenticated on startup; pairing through {chat_jid} before reporting");

        // Subscribe before asking for the code, or an authorization completed
        // while the message is still in flight arrives with nobody listening.
        let mut completions = self.codex.subscribe_login_completed().await?;
        let (url, code) = self.codex.request_device_login().await?;

        let msg = format!(
            "👋 Tera restarted and isn't paired with Codex.\n\n\
             Authorize this device:\n\
             1. Open {url}\n\
             2. Enter code: *{code}*\n\n\
             _The code expires in 15 minutes. I'll pick up where I left off once it's done._"
        );
        self.transport.send_text(chat_jid, &msg, None).await?;

        let outcome = completions.recv().await;
        // Spent either way: a retry has to be able to ask for a fresh code.
        self.codex.clear_active_login().await;

        match outcome {
            Ok(true) => {
                info!("Paired with Codex; continuing the startup report");
                Ok(())
            }
            Ok(false) => bail!("device pairing was refused or expired"),
            Err(e) => bail!("Codex stopped before pairing resolved: {e}"),
        }
    }

    fn chat_to_speak_into(&self, pending: &[ConversationTurn]) -> Result<Option<String>> {
        match pending.first() {
            Some(turn) => Ok(Some(turn.chat_jid.clone())),
            None => self.runtime_db.last_known_chat(),
        }
    }

    async fn recover(
        &self,
        chat_jid: &str,
        crashed: Option<&CrashMark>,
        update: Option<&UpdateNotice>,
        recoverable: &[ConversationTurn],
        abandoned: &[ConversationTurn],
        over_budget: bool,
    ) -> Result<()> {
        let pending_request = self.render_requests(recoverable)?;
        let abandoned_request = self.render_requests(abandoned)?;
        let restart_context =
            std::fs::read_to_string(self.config.runtime_dir().join("restart-context.md")).ok();
        let facts = serde_json::to_string_pretty(&json!({
            "time": chrono::Local::now().to_rfc3339(),
            "previous_exit": crashed,
            "update": update,
            "running_build": BuildInfo::current(),
            "recovery_disabled_by_crash_budget": over_budget,
            "restart_context": restart_context,
        }))?;

        let prompt = data::render(
            data::PHOENIX_RECOVERY_PROMPT,
            &[
                ("OWNER", &self.config.owner_name),
                ("STARTUP_FACTS", &facts),
                ("PENDING_REQUEST", &pending_request),
                ("ABANDONED_REQUEST", &abandoned_request),
            ],
        );

        let sends_before = self.session.count();
        self.session.set_chat(chat_jid);
        self.session
            .set_turn(recoverable.first().map(|turn| turn.turn_id.as_str()));
        let result = self
            .codex
            .run_task_turn(&self.config.workspace_dir, &prompt)
            .await;
        self.session.set_turn(None);
        let summary = result?;
        info!("Startup assistant finished: {summary}");

        if self.session.sends_since(sends_before) == 0 {
            if summary.trim().is_empty() {
                bail!("startup assistant produced no user-visible message for {chat_jid}");
            }
            let reply_to = recoverable
                .last()
                .or_else(|| abandoned.last())
                .map(|turn| MessageRef {
                    provider_msg_id: turn.last_provider_msg_id.clone(),
                    chat_jid: chat_jid.to_string(),
                    from_me: false,
                    text: self.quoted_text(&turn.last_provider_msg_id),
                });
            let provider_msg_id = self
                .transport
                .send_text(chat_jid, &summary, reply_to.as_ref())
                .await?;
            record_assistant_message(
                &self.history_db,
                chat_jid,
                &provider_msg_id,
                &summary,
                recoverable.first().map(|turn| turn.turn_id.clone()),
                None,
            )?;
        }
        Ok(())
    }

    fn render_requests(&self, turns: &[ConversationTurn]) -> Result<String> {
        let mut events = Vec::new();
        for turn in turns {
            events.extend(
                self.history_db
                    .messages_for_turn(&turn.turn_id)?
                    .into_iter()
                    .filter(|event| event.actor == "user"),
            );
        }
        Ok(InputRenderer::render_history(&events))
    }

    fn quoted_text(&self, provider_msg_id: &str) -> Option<String> {
        self.history_db
            .event_id_for_provider_ref("whatsapp", provider_msg_id)
            .ok()
            .flatten()
            .and_then(|event_id| self.history_db.get_event(&event_id).ok().flatten())
            .and_then(|event| event.text)
    }
}
