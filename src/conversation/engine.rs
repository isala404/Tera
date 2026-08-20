use crate::codex::process::TurnInput;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::conversation::buffer::MessageBurst;
use crate::conversation::renderer::InputRenderer;
use crate::conversation::session::ConversationSession;
use crate::conversation::typing::TypingGuard;
use crate::history::assets::AssetStorage;
use crate::conversation::record_assistant_message;
use crate::history::db::{Attachment, ConversationEvent, HistoryDb, ProviderRef};
use crate::runtime::{ActivityTracker, RuntimeDb};
use crate::secrets::{Capture, SecretStore};
use crate::transport::{InboundMessage, OwnerPolicy, Transport, Verdict};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use uuid::Uuid;

/// How long to wait for the user to stop typing before starting a turn, so a
/// three-message burst becomes one turn rather than three (PLAN.md section 13).
const BURST_QUIET_PERIOD: Duration = Duration::from_millis(2500);

/// Ceiling on the total wait from the first message of a burst. The quiet period
/// restarts on each new message, so without this someone typing steadily is never
/// answered at all (PLAN.md section 13.1).
const MAX_BURST_WAIT: Duration = Duration::from_secs(8);

/// Bursts waiting out their quiet period, and the logical turn currently being
/// executed.
///
/// One lock over both: routing an inbound message needs to consult them together,
/// and two locks taken in two orders is a deadlock waiting for a busy day.
#[derive(Default)]
struct ConversationState {
    bursts: HashMap<String, MessageBurst>,
    /// Logical turn id of the turn being executed, if any.
    running_turn: Option<String>,
}

/// What to do with a message that just arrived.
enum Route {
    /// A turn is already running: hand the message to it as it arrives, with no
    /// debounce (PLAN.md section 13.2).
    Steer,
    /// A burst is already collecting for this sender; its timer will fire.
    JoinBurst,
    /// First message of a new burst; start the quiet-period timer.
    StartBurst,
}

#[derive(Clone)]
pub struct TurnEngine {
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    transport: Arc<dyn Transport>,
    state: Arc<Mutex<ConversationState>>,
    codex: CodexSupervisor,
    owner_policy: OwnerPolicy,
    session: ConversationSession,
    activity: ActivityTracker,
    secrets: SecretStore,
}

impl TurnEngine {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        transport: Arc<dyn Transport>,
        session: ConversationSession,
        codex: CodexSupervisor,
        activity: ActivityTracker,
    ) -> Self {
        Self {
            owner_policy: OwnerPolicy::new(config.whatsapp_owner_number.clone()),
            secrets: SecretStore::new(config.secrets_path()),
            config,
            history_db,
            runtime_db,
            transport,
            state: Arc::new(Mutex::new(ConversationState::default())),
            codex,
            session,
            activity,
        }
    }

    pub async fn handle_inbound_message(&self, msg: InboundMessage) -> Result<()> {
        // Gate before anything is recorded or executed. Codex runs here with no
        // approvals and full disk access, so a stranger's message must not reach
        // it, and must not pollute the owner's conversation history either.
        if let Verdict::Reject(reason) = self.owner_policy.evaluate(&msg) {
            warn!(
                "Ignoring message from {}: {}. If this is you, set WHATSAPP_OWNER_JID={}",
                msg.sender,
                reason,
                crate::transport::owner::jid_user(&msg.sender)
            );
            return Ok(());
        }

        // Before the message is recorded, executed or even looked at: if it
        // carried a credential, take the value out and leave a note in its place.
        // Everything downstream, history, the JSONL projection, the thread and
        // every memory generation built from it, stores whatever survives here.
        let msg = self.capture_secret(msg);

        let sender = msg.sender.clone();
        let event_id = format!("msg_{}", Uuid::new_v4().simple());

        // Tool calls made during this turn reply into the chat, not the sending
        // device: a device-suffixed JID is not a valid reaction target.
        self.session.set_chat(&msg.chat_jid);

        // 1. Persist any attachment before recording the event, so history never
        //    references an asset that is not on disk.
        let attachments = self.persist_media(&msg, &event_id)?;

        // 2. Decide where this message goes before recording it, because the
        //    answer determines the logical turn id it is stamped with. Without
        //    that, user messages had no `turn` in history at all and a past
        //    exchange could not be reconstructed (PLAN.md section 13.3).
        let (route, logical_turn) = self.route(&sender).await;

        let conv_ev = ConversationEvent {
            seq: None,
            id: event_id.clone(),
            occurred_at_ms: msg.timestamp_ms,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: msg.text.clone(),
            reply_to_id: self.resolve_reply_target(msg.reply_to_provider_msg_id.as_deref()),
            turn_id: Some(logical_turn.clone()),
            reaction_target_id: None,
            reaction_emoji: None,
            attachments,
        };

        self.history_db.insert_event(conv_ev.clone())?;
        self.history_db
            .record_provider_ref(&ProviderRef::whatsapp(
                &event_id,
                &msg.provider_msg_id,
                &msg.chat_jid,
                msg.from_own_account,
            ))?;

        info!("Recorded inbound message from {}: {:?}", sender, msg.text);

        match route {
            Route::Steer => {
                let inputs = self.turn_inputs(std::slice::from_ref(&conv_ev));
                if self.codex.steer_main_turn(&inputs).await? {
                    return Ok(());
                }
                // The turn finished in the gap. Fall through and treat this as the
                // start of a new one rather than dropping the message.
                info!("Nothing to steer after all; starting a new turn for this message");
                self.begin_burst(&sender, &msg.chat_jid, conv_ev, &msg.provider_msg_id)
                    .await;
            }
            Route::JoinBurst => {
                let mut state = self.state.lock().await;
                if let Some(burst) = state.bursts.get_mut(&sender) {
                    burst.push(conv_ev);
                } else {
                    // Its timer fired while we were writing to history.
                    drop(state);
                    self.begin_burst(&sender, &msg.chat_jid, conv_ev, &msg.provider_msg_id)
                        .await;
                }
            }
            Route::StartBurst => {
                self.begin_burst(&sender, &msg.chat_jid, conv_ev, &msg.provider_msg_id)
                    .await;
            }
        }

        Ok(())
    }

    /// Swap a credential out of an inbound message for a note about it.
    ///
    /// The owner has one channel to this daemon and it feeds a model, so a key
    /// typed into the chat would otherwise be permanent context. Substitution
    /// rather than a second code path: the message still routes, still starts a
    /// turn, still steers a running one. Only its text is different, so the agent
    /// learns the credential arrived and can carry on with whatever needed it.
    ///
    /// See [`crate::secrets`] for what this does and does not protect against.
    fn capture_secret(&self, mut msg: InboundMessage) -> InboundMessage {
        let Some(text) = msg.text.as_deref() else {
            return msg;
        };

        let outcome = match self.secrets.capture(text, msg.timestamp_ms) {
            Ok(outcome) => outcome,
            Err(error) => {
                // The store is unreadable, so whether a request was armed is
                // unknown. An explicit command is the one case where the message
                // is known to hold a credential, so that must not fall through;
                // swallowing every other message over a bad file would turn a
                // broken extra into a broken daemon.
                error!("Cannot reach the secret store: {error:?}");
                if text.trim_start().starts_with("/secret") {
                    Capture::Rejected {
                        reason: format!("the secret store could not be read: {error}"),
                    }
                } else {
                    Capture::Passthrough
                }
            }
        };

        let note = match outcome {
            Capture::Passthrough => return msg,
            Capture::Stored { name } => {
                info!("Captured secret {name} from an inbound message");
                format!(
                    "[Tera took this message before you saw it, because it carried a credential. \
                     {name} is stored now, and skills read it from the secret store. You cannot see \
                     the value and asking for it will not work. Tell {owner} it is saved, ask them \
                     to delete the message they just sent from this chat, and carry on with \
                     whatever needed it.]",
                    owner = self.config.owner_name
                )
            }
            Capture::Rejected { reason } => {
                warn!("Discarded a secret-bearing message: {reason}");
                format!(
                    "[Tera took this message before you saw it, because it looked like a \
                     credential. It could not be stored: {reason}. The text was discarded either \
                     way. Tell {owner} what went wrong and ask them to send it again.]",
                    owner = self.config.owner_name
                )
            }
        };

        msg.text = Some(note);
        // A message carrying a credential has no business also carrying a file
        // into history, and a screenshot of a dashboard is exactly what would
        // arrive here.
        msg.media_attachment = None;
        msg
    }

    /// Translate a WhatsApp reply target into our own event id.
    ///
    /// `None` when the message is not a reply, or replies to something older than
    /// this workspace. Better an absent field than a `reply_to` pointing at an id
    /// that appears nowhere in history.
    fn resolve_reply_target(&self, provider_msg_id: Option<&str>) -> Option<String> {
        let provider_msg_id = provider_msg_id?;
        match self
            .history_db
            .event_id_for_provider_ref("whatsapp", provider_msg_id)
        {
            Ok(Some(event_id)) => Some(event_id),
            Ok(None) => {
                info!("Reply target {provider_msg_id} is not in history; recording it without one");
                None
            }
            Err(e) => {
                warn!("Could not resolve reply target {provider_msg_id}: {e:?}");
                None
            }
        }
    }

    /// Where an inbound message belongs, and under which logical turn id.
    async fn route(&self, sender: &str) -> (Route, String) {
        let state = self.state.lock().await;

        // Only steer when a turn is genuinely in flight on the app-server. The
        // engine's own view can lag behind a turn that just completed.
        if let Some(running) = state.running_turn.clone() {
            drop(state);
            if self.codex.main_turn_is_running().await {
                return (Route::Steer, running);
            }
            return (Route::StartBurst, format!("turn_{}", Uuid::new_v4().simple()));
        }

        match state.bursts.get(sender) {
            Some(burst) => (Route::JoinBurst, burst.turn_id.clone()),
            None => (
                Route::StartBurst,
                format!("turn_{}", Uuid::new_v4().simple()),
            ),
        }
    }

    /// Open a burst for `sender` and arm its quiet-period timer.
    async fn begin_burst(
        &self,
        sender: &str,
        chat_jid: &str,
        event: ConversationEvent,
        last_provider_msg_id: &str,
    ) {
        let turn_id = event
            .turn_id
            .clone()
            .unwrap_or_else(|| format!("turn_{}", Uuid::new_v4().simple()));

        // Durable before the quiet period, not after: a crash while buffering
        // still leaves a turn Phoenix can see and answer.
        if let Err(e) = self
            .runtime_db
            .start_turn(&turn_id, chat_jid, last_provider_msg_id)
        {
            warn!("Could not record turn {turn_id}; a crash now would lose it: {e:?}");
        }

        {
            let mut state = self.state.lock().await;
            state
                .bursts
                .insert(sender.to_string(), MessageBurst::new(turn_id, event));
        }

        let engine = self.clone();
        let sender = sender.to_string();
        let last_provider_msg_id = last_provider_msg_id.to_string();

        tokio::spawn(async move {
            // Wait out the quiet period, restarting it whenever another message
            // lands, but never past the ceiling: someone typing continuously
            // still gets an answer.
            loop {
                let remaining = {
                    let state = engine.state.lock().await;
                    match state.bursts.get(&sender) {
                        Some(burst) => burst.remaining_wait(BURST_QUIET_PERIOD, MAX_BURST_WAIT),
                        // Something else already took it.
                        None => return,
                    }
                };

                if remaining.is_zero() {
                    break;
                }
                tokio::time::sleep(remaining).await;
            }

            let burst_opt = {
                let mut state = engine.state.lock().await;
                state.bursts.remove(&sender)
            };

            if let Some(burst) = burst_opt {
                if let Err(e) = engine
                    .process_burst(&sender, burst, &last_provider_msg_id)
                    .await
                {
                    error!("Failed to process burst for {}: {:?}", sender, e);
                }
            }
        });
    }

    /// Write inbound media into the asset store and describe it as an
    /// attachment row. Originals are kept byte-for-byte (PLAN.md section 19).
    fn persist_media(&self, msg: &InboundMessage, event_id: &str) -> Result<Vec<Attachment>> {
        let Some(media) = &msg.media_attachment else {
            return Ok(vec![]);
        };

        let (_full, relative_path) = AssetStorage::save_attachment(
            &self.config,
            event_id,
            msg.timestamp_ms,
            &media.filename,
            &media.data,
        )?;

        info!("Stored {} attachment at {}", media.media_type, relative_path);

        Ok(vec![Attachment {
            id: None,
            event_id: event_id.to_string(),
            position: 0,
            media_type: media.media_type.clone(),
            relative_path,
            mime_type: Some(media.mime_type.clone()),
            original_name: Some(media.filename.clone()),
        }])
    }

    /// Render events into the turn input Codex receives: the text, then any media
    /// it can read natively.
    fn turn_inputs(&self, events: &[ConversationEvent]) -> Vec<TurnInput> {
        let mut inputs = vec![TurnInput::Text(InputRenderer::render_burst(events))];
        inputs.extend(self.media_inputs(events));
        inputs
    }

    /// Media in these events, as Codex turn inputs.
    ///
    /// Only images and audio are passed natively. Those are the modalities the
    /// app-server accepts. Video, documents and stickers stay described in the
    /// rendered text, where the agent can still read the path off disk.
    fn media_inputs(&self, events: &[ConversationEvent]) -> Vec<TurnInput> {
        events
            .iter()
            .flat_map(|event| &event.attachments)
            .filter_map(|att| {
                let path = self.config.resolve_asset(&att.relative_path);
                match att.media_type.as_str() {
                    "image" | "sticker" => Some(TurnInput::LocalImage(path)),
                    "audio" => Some(TurnInput::LocalAudio(path)),
                    _ => None,
                }
            })
            .collect()
    }

    /// Run a burst as one turn, with the "a turn is running" flag held for
    /// exactly its duration.
    ///
    /// Marking and clearing live here, in one place, rather than at each exit
    /// path inside the turn body. There are several, and one that forgot to
    /// clear would wedge every later message into steering a turn that ended.
    async fn process_burst(
        &self,
        sender: &str,
        burst: MessageBurst,
        last_provider_msg_id: &str,
    ) -> Result<()> {
        // Registers the conversation as busy for the duration, which is what
        // defers memory maintenance and interrupts it if it is already running.
        let _active = self.activity.begin();

        let turn_id = burst.turn_id.clone();
        self.set_running(Some(turn_id.clone())).await;
        let outcome = self.execute_turn(sender, burst, last_provider_msg_id).await;
        self.set_running(None).await;

        // Closed either way. This process is still alive, so a failure here is a
        // logged failure, not something for Phoenix to resurrect at a restart
        // that might be days away.
        let state = if outcome.is_ok() { "completed" } else { "failed" };
        if let Err(e) = self.runtime_db.finish_turn(&turn_id, state) {
            warn!("Could not close turn {turn_id}: {e:?}");
        }
        outcome
    }

    async fn set_running(&self, turn: Option<String>) {
        self.state.lock().await.running_turn = turn;
    }

    async fn execute_turn(
        &self,
        sender: &str,
        burst: MessageBurst,
        last_provider_msg_id: &str,
    ) -> Result<()> {
        info!("Processing message burst for {} ({} events)", sender, burst.events.len());

        // Snapshot before the turn so send_message calls made during it are visible.
        let sends_before = self.session.count();

        // Held for the whole turn; clears itself however this function exits.
        let _typing = TypingGuard::start(self.transport.clone(), sender.to_string());
        let chat_jid = self.session.chat().unwrap_or_default();
        self.session.set_turn(Some(&burst.turn_id));

        let result = async {
            // Render events into a structured prompt, then hand any images and voice
            // notes to Codex as real media rather than a text description of media.
            let inputs = self.turn_inputs(&burst.events);

            // A degraded turn must read as degraded. Echoing a canned "I'm ready to
            // assist" makes a dead Codex backend indistinguishable from a real reply.
            let reply_text = match self.codex.run_main_turn(&inputs).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Codex turn failed: {:?}", e);
                    format!("⚠️ I couldn't complete that turn, the Codex backend errored: {e}")
                }
            };

            self.codex.note_main_activity();

            // The agent is instructed to reply through the send_message MCP tool, and
            // usually does. Sending the final agent text unconditionally would then
            // deliver every answer twice. Only fall back when the turn produced no
            // user-visible message of its own (PLAN.md section 54.1).
            if self.session.sends_since(sends_before) > 0 {
                info!("Turn replied via send_message; skipping final-text fallback");
                return Ok(());
            }

            // Nothing was sent and there is nothing to send: the user asked something
            // and would otherwise get an answer, so say so rather than leave them waiting.
            let reply_text = if reply_text.trim().is_empty() {
                warn!("Turn produced neither a send_message nor any final text");
                "⚠️ I finished working on that but didn't produce a reply. Try asking again."
                    .to_string()
            } else {
                reply_text
            };

            let outbound_msg_id = self
                .transport
                .send_text(sender, &reply_text, Some(last_provider_msg_id))
                .await?;

            record_assistant_message(
                &self.history_db,
                &chat_jid,
                &outbound_msg_id,
                &reply_text,
                Some(burst.turn_id.clone()),
                burst.events.last().map(|e| e.id.clone()),
            )?;

            info!("Sent reply to {}: {}", sender, reply_text);
            Ok(())
        }
        .await;

        self.session.set_turn(None);
        result
    }
}
