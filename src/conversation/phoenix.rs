//! What runs when the last life ended badly.
//!
//! Phoenix has one job with two halves: tell the owner what happened, then try
//! to put things back. It fires on two signals, both read at startup — a crash
//! mark the previous process left behind, and any conversation turn that was
//! accepted but never answered.
//!
//! Two rules shape the rest of it:
//!
//! * **The telling must not depend on what broke.** The first message goes out
//!   through the transport alone: no Codex, no workspace, no MCP. Whatever else
//!   is wrong, the owner finds out.
//! * **Recovery gets a fresh thread, never the conversation's own.** If the
//!   crash came out of that thread's state, resuming it recreates the crash.
//!   Phoenix reads what was asked and treats it as new work with context.

use crate::codex::tier;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::conversation::record_assistant_message;
use crate::conversation::renderer::InputRenderer;
use crate::data;
use crate::history::db::HistoryDb;
use crate::runtime::phoenix::CrashMark;
use crate::runtime::{ActivityTracker, ConversationTurn, RuntimeDb};
use crate::transport::Transport;
use crate::update::UpdateNotice;
use anyhow::Result;
use std::sync::Arc;
use tracing::{info, warn};

/// Recovery attempts a single turn gets before it is abandoned.
///
/// The poison pill is usually one specific request, so the budget belongs on the
/// turn rather than on the boot. Two tries, then say so and stop.
const MAX_TURN_ATTEMPTS: i64 = 2;

/// Unclean starts in a row before Phoenix stops trying to fix anything.
///
/// An agent that just failed to repair the system will not repair it now, and a
/// crash loop that also sends messages is worse than one that stays quiet.
const MAX_CONSECUTIVE_CRASHES: u32 = 2;

pub struct Phoenix {
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    transport: Arc<dyn Transport>,
    codex: CodexSupervisor,
    activity: ActivityTracker,
}

impl Phoenix {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        transport: Arc<dyn Transport>,
        codex: CodexSupervisor,
        activity: ActivityTracker,
    ) -> Self {
        Self {
            config,
            history_db,
            runtime_db,
            transport,
            codex,
            activity,
        }
    }

    /// Report and recover. Called once, at startup, with whatever the previous
    /// process left behind.
    ///
    /// Nothing is written until the owner has actually been told, so a failure
    /// to reach them — the transport is usually still connecting at this point —
    /// leaves the whole job untouched and safe to retry.
    pub async fn run(
        &self,
        crashed: Option<CrashMark>,
        update: Option<UpdateNotice>,
    ) -> Result<()> {
        let pending = self.runtime_db.unfinished_turns()?;
        if crashed.is_none() && pending.is_empty() && update.is_none() {
            return Ok(());
        }

        let Some(chat_jid) = self.chat_to_speak_into(&pending)? else {
            warn!("Phoenix has something to report but no conversation to report it in");
            return Ok(());
        };

        let over_budget = crashed
            .as_ref()
            .is_some_and(|c| c.consecutive >= MAX_CONSECUTIVE_CRASHES);
        let (recoverable, abandoned): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|turn| !over_budget && turn.attempts < MAX_TURN_ATTEMPTS);

        self.report(
            &chat_jid,
            crashed.as_ref(),
            update.as_ref(),
            &recoverable,
            &abandoned,
        )
            .await?;

        for turn in recoverable.iter().chain(&abandoned) {
            self.runtime_db.record_turn_attempt(&turn.turn_id)?;
        }
        for turn in &abandoned {
            warn!("Phoenix abandoned turn {} after {} attempts", turn.turn_id, turn.attempts + 1);
            self.runtime_db.finish_turn(&turn.turn_id, "abandoned")?;
        }

        if over_budget {
            warn!("Phoenix is over its crash budget; reporting only, no repair attempted");
            return Ok(());
        }

        if crashed.is_none() && recoverable.is_empty() {
            return Ok(());
        }

        self.repair(&chat_jid, crashed.as_ref(), recoverable).await
    }

    /// Where to speak. An interrupted turn names its own chat; otherwise the
    /// last conversation this daemon had is the best answer available.
    fn chat_to_speak_into(&self, pending: &[ConversationTurn]) -> Result<Option<String>> {
        match pending.first() {
            Some(turn) => Ok(Some(turn.chat_jid.clone())),
            None => self.runtime_db.last_known_chat(),
        }
    }

    /// The one message that must get through. Plain transport, no agent: this
    /// runs before anything that could still be broken is touched.
    async fn report(
        &self,
        chat_jid: &str,
        crashed: Option<&CrashMark>,
        update: Option<&UpdateNotice>,
        recoverable: &[ConversationTurn],
        abandoned: &[ConversationTurn],
    ) -> Result<()> {
        let mut lines = Vec::new();
        if let Some(mark) = crashed {
            lines.push(format!("I {}. Restarted now.", mark.describe()));
        } else if !recoverable.is_empty() || !abandoned.is_empty() {
            lines.push("I restarted while I was still working on something.".to_string());
        }
        if let Some(update) = update {
            lines.push(update.message());
        }

        if !recoverable.is_empty() {
            lines.push("You were waiting on me. Picking that back up.".to_string());
        }
        if !abandoned.is_empty() {
            lines.push(
                "One thing I can't retry: it has already crashed me twice, so I'm leaving it. Ask again if you still want it.".to_string(),
            );
        }
        if crashed.is_some() && recoverable.is_empty() && abandoned.is_empty() {
            lines.push("Nothing of yours was in flight. Checking myself over.".to_string());
        }

        let text = lines.join(" ");
        let reply_to = recoverable
            .first()
            .or(abandoned.first())
            .map(|t| t.last_provider_msg_id.clone());

        let provider_msg_id = self
            .transport
            .send_text(chat_jid, &text, reply_to.as_deref())
            .await?;
        record_assistant_message(
            &self.history_db,
            chat_jid,
            &provider_msg_id,
            &text,
            None,
            None,
        )?;
        info!("Phoenix reported to {chat_jid}: {text}");
        Ok(())
    }

    /// One isolated thread, rooted in the workspace, that finishes what was
    /// interrupted and checks the system over. Not the conversation's thread.
    async fn repair(
        &self,
        chat_jid: &str,
        crashed: Option<&CrashMark>,
        recoverable: Vec<ConversationTurn>,
    ) -> Result<()> {
        let _active = self.activity.begin();

        let pending_request = if recoverable.is_empty() {
            "Nothing of theirs was in flight.".to_string()
        } else {
            InputRenderer::render_history(&self.history_db.recent_messages(10)?)
        };

        let prompt = data::render(
            data::PHOENIX_RECOVERY_PROMPT,
            &[
                ("OWNER", &self.config.owner_name),
                ("NOW", &chrono::Local::now().to_rfc3339()),
                (
                    "WHAT_HAPPENED",
                    &crashed
                        .map(|c| c.describe())
                        .unwrap_or_else(|| "restarted mid-turn".to_string()),
                ),
                ("PENDING_REQUEST", &pending_request),
            ],
        );

        let summary = self
            .codex
            .run_task_turn(&self.config.workspace_dir, &prompt, tier::CONVERSATION)
            .await?;
        info!("Phoenix recovery finished: {summary}");

        // Only now are the turns answered. An error above leaves them open, so
        // the next start tries again, bounded by the attempt count already spent.
        for turn in &recoverable {
            self.runtime_db.finish_turn(&turn.turn_id, "completed")?;
        }

        // The agent replies through `send_message`; a run that produced nothing
        // visible would otherwise end the recovery in silence.
        if !recoverable.is_empty() && summary.trim().is_empty() {
            warn!("Phoenix recovery produced no reply for {chat_jid}");
        }
        Ok(())
    }
}
