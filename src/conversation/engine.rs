use crate::codex::process::TurnInput;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::conversation::buffer::MessageBurst;
use crate::conversation::record_assistant_message;
use crate::conversation::renderer::InputRenderer;
use crate::conversation::session::ConversationSession;
use crate::conversation::typing::TypingGuard;
use crate::history::assets::AssetStorage;
use crate::history::db::{Attachment, ConversationEvent, HistoryDb, ProviderRef};
use crate::runtime::{RuntimeDb, TurnState};
use crate::secrets::{Capture, SecretStore};
use crate::transport::owner::jid_user;
use crate::transport::{
    InboundMessage, InboundPresence, InboundPresenceKind, MessageRef, OwnerPolicy, Transport,
    Verdict,
};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// How long to wait for the user to stop typing before starting a turn, so a
/// three-message burst becomes one turn rather than three.
const BURST_QUIET_PERIOD: Duration = Duration::from_millis(2500);

/// Ceiling on the total wait from the first message of a burst. The quiet period
/// restarts on each new message, so without this someone typing steadily is never
/// answered at all.
const MAX_BURST_WAIT: Duration = Duration::from_secs(8);

/// Presence events do not carry a duration. Poll while one is active, and stop
/// trusting it eventually in case WhatsApp disconnects before sending Paused.
const PRESENCE_POLL_INTERVAL: Duration = Duration::from_millis(500);
const MAX_PRESENCE_HOLD: Duration = Duration::from_secs(5 * 60);

const MODEL_FAILURE_REPLY: &str =
    "I couldn't complete that because the model provider is unavailable. Check Tera's service log for the cause, then try again.";

/// Bursts waiting out their quiet period, and the logical turn currently being
/// executed.
///
/// One lock over both: routing an inbound message needs to consult them together,
/// and two locks taken in two orders is a deadlock waiting for a busy day.
#[derive(Default)]
struct ConversationState {
    bursts: HashMap<String, MessageBurst>,
    composing_since: HashMap<String, Instant>,
    /// Logical turn id of the turn being executed, if any.
    running_turn: Option<String>,
    in_flight_provider_msgs: HashSet<String>,
}

impl ConversationState {
    fn update_presence(&mut self, sender: &str, kind: InboundPresenceKind) {
        let user = jid_user(sender).to_string();
        match kind {
            InboundPresenceKind::Typing | InboundPresenceKind::RecordingAudio => {
                self.composing_since
                    .entry(user)
                    .or_insert_with(Instant::now);
            }
            InboundPresenceKind::Paused => {
                self.composing_since.remove(&user);
                for (burst_sender, burst) in &mut self.bursts {
                    if jid_user(burst_sender) == user {
                        burst.restart_quiet_period();
                    }
                }
            }
        }
    }

    fn remaining_wait(&mut self, sender: &str) -> Option<Duration> {
        let user = jid_user(sender);
        let composing = self
            .composing_since
            .get(user)
            .is_some_and(|started| started.elapsed() < MAX_PRESENCE_HOLD);
        if self.composing_since.contains_key(user) && !composing {
            self.composing_since.remove(user);
        }

        let burst = self.bursts.get(sender)?;
        if composing {
            let deadline = MAX_BURST_WAIT.saturating_sub(burst.created_at.elapsed());
            if deadline.is_zero() {
                self.composing_since.remove(user);
            }
            return Some(PRESENCE_POLL_INTERVAL.min(deadline));
        }

        Some(burst.remaining_wait(BURST_QUIET_PERIOD, MAX_BURST_WAIT))
    }
}

/// What to do with a message that just arrived.
enum Route {
    /// A turn is already running: hand the message to it as it arrives, with no
    /// debounce.
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
    secrets: SecretStore,
    login_notifier_active: Arc<std::sync::atomic::AtomicBool>,
}

impl TurnEngine {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        transport: Arc<dyn Transport>,
        session: ConversationSession,
        codex: CodexSupervisor,
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
            login_notifier_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

        // Nothing to answer means nothing to start a turn for, and nothing worth
        // a row in history either. Dropping these is what keeps a history sync
        // replay from opening a turn on the paired account's own chat.
        if !msg.is_answerable() {
            debug!(
                "Ignoring message {} from {}: no text and no attachment",
                msg.provider_msg_id, msg.sender
            );
            return Ok(());
        }

        // Deduplicate in-flight concurrent deliveries of the same message
        {
            let mut state = self.state.lock().await;
            if !state
                .in_flight_provider_msgs
                .insert(msg.provider_msg_id.clone())
            {
                info!(
                    "Dropping in-flight duplicate message {}",
                    msg.provider_msg_id
                );
                return Ok(());
            }
        }

        struct InFlightGuard {
            state: Arc<Mutex<ConversationState>>,
            id: String,
        }
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                let state = self.state.clone();
                let id = self.id.clone();
                tokio::spawn(async move {
                    state.lock().await.in_flight_provider_msgs.remove(&id);
                });
            }
        }
        let _guard = InFlightGuard {
            state: self.state.clone(),
            id: msg.provider_msg_id.clone(),
        };

        // Deduplicate against already committed messages before secret capture or turn routing
        if self
            .history_db
            .is_provider_message_recorded("whatsapp", &msg.provider_msg_id)?
        {
            info!("Dropping already recorded message {}", msg.provider_msg_id);
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
        //    exchange could not be reconstructed.
        let (route, logical_turn) = self.route(&sender).await;

        let effective_text = match (msg.text.as_deref(), msg.media_error.as_deref()) {
            (Some(caption), Some(err)) if !caption.trim().is_empty() => {
                Some(format!("{caption}\n\n[Attachment download failed: {err}]"))
            }
            (_, Some(err)) => Some(format!("[Attachment download failed: {err}]")),
            (Some(caption), None) => Some(caption.to_string()),
            (None, None) => None,
        };

        let conv_ev = ConversationEvent::message(
            event_id.clone(),
            msg.timestamp_ms,
            "user",
            effective_text,
            self.resolve_reply_target(msg.reply_to_provider_msg_id.as_deref()),
            Some(logical_turn.clone()),
            attachments,
        );

        let saved = self.history_db.insert_inbound_event(
            conv_ev.clone(),
            ProviderRef::whatsapp(
                &event_id,
                &msg.provider_msg_id,
                &msg.chat_jid,
                msg.from_own_account,
            ),
        )?;

        if saved.is_none() {
            info!(
                "Inbound message {} deduplicated at commit",
                msg.provider_msg_id
            );
            return Ok(());
        }

        info!("Recorded inbound message from {}: {:?}", sender, msg.text);

        if let Some(ref media_err) = msg.media_error {
            let failure_reply = format!("⚠️ Could not download attachment: {media_err}");
            let reply_target = MessageRef {
                provider_msg_id: msg.provider_msg_id.clone(),
                chat_jid: msg.chat_jid.clone(),
                from_me: msg.from_own_account,
                text: msg.text.clone(),
            };
            let outbound_msg_id = self
                .transport
                .send_text(&msg.chat_jid, &failure_reply, Some(&reply_target))
                .await?;
            record_assistant_message(
                &self.history_db,
                &msg.chat_jid,
                &outbound_msg_id,
                &failure_reply,
                Some(logical_turn.clone()),
                Some(event_id.clone()),
            )?;

            // If file-only, there is no user text to answer, so the turn ends here.
            if msg.text.as_deref().is_none_or(|t| t.trim().is_empty()) {
                return Ok(());
            }
        }

        match route {
            Route::Steer => {
                let inputs = self.turn_inputs(std::slice::from_ref(&conv_ev));
                if self.codex.steer_main_turn(&inputs).await? {
                    let _ = self
                        .runtime_db
                        .update_turn_last_provider_msg_id(&logical_turn, &msg.provider_msg_id);
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
                    let turn_id = burst.turn_id.clone();
                    burst.push(conv_ev);
                    let _ = self
                        .runtime_db
                        .update_turn_last_provider_msg_id(&turn_id, &msg.provider_msg_id);
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

    /// Keep a buffered turn open while the owner is still typing or recording.
    /// Presence is advisory and never enters history or reaches Codex.
    pub async fn handle_presence(&self, presence: InboundPresence) {
        if let Verdict::Reject(_) = self.owner_policy.evaluate_sender(
            &presence.sender,
            presence.from_own_account,
            presence.is_group,
        ) {
            return;
        }

        match presence.kind {
            InboundPresenceKind::Typing => info!("Owner is typing"),
            InboundPresenceKind::RecordingAudio => info!("Owner is recording audio"),
            InboundPresenceKind::Paused => info!("Owner stopped composing"),
        }
        self.state
            .lock()
            .await
            .update_presence(&presence.sender, presence.kind);
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
                error!("Cannot reach the secret store: {error:?}");
                Capture::Rejected {
                    reason: format!("the secret store could not be read: {error}"),
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
            return (
                Route::StartBurst,
                format!("turn_{}", Uuid::new_v4().simple()),
            );
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
            .turn_id()
            .map(str::to_string)
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
        let typing_recipient = chat_jid.to_string();
        let last_provider_msg_id = last_provider_msg_id.to_string();

        tokio::spawn(async move {
            // Acknowledge the accepted message immediately. The guard stays live
            // through buffering and execution, and always addresses the chat
            // rather than a device-suffixed sender JID.
            let _typing = TypingGuard::start(engine.transport.clone(), typing_recipient);

            // Wait out the quiet period, restarting it whenever another message
            // lands, but never past the ceiling: someone typing continuously
            // still gets an answer.
            loop {
                let remaining = {
                    let mut state = engine.state.lock().await;
                    match state.remaining_wait(&sender) {
                        Some(remaining) => remaining,
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
    /// attachment row. Originals are kept byte-for-byte.
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

        info!(
            "Stored {} attachment at {}",
            media.media_type, relative_path
        );

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
        let mut reply_targets = HashMap::new();
        for event in events {
            let Some(reply_to) = event.reply_to_id() else {
                continue;
            };
            match self.history_db.get_event(reply_to) {
                Ok(Some(target)) => {
                    reply_targets.insert(target.id.clone(), target);
                }
                Ok(None) => {
                    info!("Reply target {reply_to} is not available for prompt injection");
                }
                Err(e) => {
                    warn!("Could not load reply target {reply_to} for prompt injection: {e:?}");
                }
            }
        }

        let mut inputs = vec![TurnInput::Text(InputRenderer::render_burst_with_replies(
            events,
            &reply_targets,
        ))];
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
        // Registers the conversation turn as busy for the duration.

        let turn_id = burst.turn_id.clone();
        self.set_running(Some(turn_id.clone())).await;
        let outcome = self.execute_turn(sender, burst, last_provider_msg_id).await;
        self.set_running(None).await;

        // Closed either way. This process is still alive, so a failure here is a
        // logged failure, not something for Phoenix to resurrect at a restart
        // that might be days away.
        let state = if outcome.is_ok() {
            TurnState::Completed
        } else {
            TurnState::Failed
        };
        if let Err(e) = self.runtime_db.finish_turn(&turn_id, state) {
            warn!("Could not close turn {turn_id}: {e:?}");
        }
        outcome
    }

    async fn set_running(&self, turn: Option<String>) {
        self.state.lock().await.running_turn = turn;
    }

    /// Run the turn.
    ///
    /// Outside errors (transport down, Codex unauthenticated) are reported into
    /// the chat rather than abandoned, so the owner sees what happened.
    async fn execute_turn(
        &self,
        sender: &str,
        burst: MessageBurst,
        last_provider_msg_id: &str,
    ) -> Result<()> {
        info!(
            "Processing message burst for {} ({} events)",
            sender,
            burst.events.len()
        );

        // Snapshot before the turn so send_message calls made during it are visible.
        let sends_before = self.session.count_for(Some("foreground"));

        // A chat, never `sender`: that is a device-suffixed JID, which is not a
        // routable address. The session was set from this message's chat before
        // the burst opened.
        let chat_jid = self.session.chat().unwrap_or_default();
        self.session
            .set_turn_for("foreground", Some(&burst.turn_id));

        let result = async {
            // Render events into a structured prompt, then hand any images and voice
            // notes to Codex as real media rather than a text description of media.
            let inputs = self.turn_inputs(&burst.events);

            let reply_target = if let Some(last) = burst.events.last() {
                let stored_ref = self
                    .history_db
                    .lookup_provider_ref_by_event_id(&last.id, "whatsapp")
                    .ok()
                    .flatten();
                Some(MessageRef {
                    provider_msg_id: stored_ref
                        .as_ref()
                        .map(|r| r.provider_msg_id.clone())
                        .unwrap_or_else(|| last_provider_msg_id.to_string()),
                    chat_jid: stored_ref
                        .as_ref()
                        .filter(|r| !r.chat_jid.is_empty())
                        .map(|r| r.chat_jid.clone())
                        .unwrap_or_else(|| chat_jid.clone()),
                    from_me: stored_ref.as_ref().map(|r| r.from_me).unwrap_or(false),
                    text: last.text().map(str::to_string),
                })
            } else {
                None
            };

            if !self.codex.check_authenticated().await {
                info!("Codex unauthenticated; sending device pairing instructions to {chat_jid}");

                // Subscribe before asking for the code. A completion that landed
                // between the request and the subscription would be lost, and the
                // owner would sit waiting for a confirmation that never comes.
                let completions = self.codex.subscribe_login_completed().await;

                match self.codex.request_device_login().await {
                    Ok((url, code)) => {
                        // This turn ends here: waiting out the pairing would hold
                        // the conversation open for up to fifteen minutes, and
                        // every message sent meanwhile would steer into a turn
                        // that cannot run. So say plainly that the question needs
                        // sending again.
                        let prompt = format!(
                            "👋 I'm not paired with Codex yet, so I can't answer that.\n\n\
                             Authorize this device:\n\
                             1. Open {url}\n\
                             2. Enter code: *{code}*\n\n\
                             _The code expires in 15 minutes. I'll tell you when it's done, then send your message again._"
                        );
                        let outbound_msg_id = self
                            .transport
                            .send_text(&chat_jid, &prompt, reply_target.as_ref())
                            .await?;
                        record_assistant_message(
                            &self.history_db,
                            &chat_jid,
                            &outbound_msg_id,
                            &prompt,
                            Some(burst.turn_id.clone()),
                            burst.events.last().map(|e| e.id.clone()),
                        )?;
                        match completions {
                            Ok(rx) => self.spawn_login_notifier(chat_jid.clone(), rx),
                            Err(e) => warn!(
                                "Pairing code sent, but login completions are unreadable so the owner will not be told: {e:?}"
                            ),
                        }
                        return Ok(());
                    }
                    Err(e) => {
                        error!("Failed to request device login from codex app-server: {e:?}");
                    }
                }
            }

            let reply_text = match self.codex.run_main_turn(&inputs).await {
                Ok(reply) => reply,
                Err(error) => {
                    error!("Codex turn failed: {error:?}");
                    if self.session.sends_since_for(Some("foreground"), sends_before) == 0 {
                        match self
                            .transport
                            .send_text(&chat_jid, MODEL_FAILURE_REPLY, reply_target.as_ref())
                            .await
                        {
                            Ok(outbound_msg_id) => {
                                if let Err(record_error) = record_assistant_message(
                                    &self.history_db,
                                    &chat_jid,
                                    &outbound_msg_id,
                                    MODEL_FAILURE_REPLY,
                                    Some(burst.turn_id.clone()),
                                    burst.events.last().map(|event| event.id.clone()),
                                ) {
                                    warn!("Could not record model failure reply: {record_error:?}");
                                }
                            }
                            Err(send_error) => {
                                error!("Could not send model failure reply: {send_error:?}");
                            }
                        }
                    }
                    return Err(error);
                }
            };

            self.codex.note_main_activity();

            // The agent is instructed to reply through the send_message MCP tool, and
            // usually does. Sending the final agent text unconditionally would then
            // deliver every answer twice. Only fall back when the turn produced no
            // user-visible message of its own.
            if self.session.sends_since_for(Some("foreground"), sends_before) > 0 {
                info!("Turn replied via send_message; skipping final-text fallback");
                return Ok(());
            }

            if reply_text.trim().is_empty() {
                warn!("Turn produced neither a send_message nor any final text");
                anyhow::bail!("turn produced no user-visible reply");
            }

            // The fallback answers the burst, so it quotes the message that
            // closed it (including any steering message that arrived during the turn).
            let latest_provider_msg_id = self
                .runtime_db
                .get_turn(&burst.turn_id)
                .ok()
                .flatten()
                .map(|t| t.last_provider_msg_id)
                .unwrap_or_else(|| last_provider_msg_id.to_string());

            let reply_target = {
                let stored_ref = self
                    .history_db
                    .lookup_provider_ref_by_provider_id(&latest_provider_msg_id, "whatsapp")
                    .ok()
                    .flatten();
                let last_event = self
                    .history_db
                    .list_turn_events(&burst.turn_id)
                    .ok()
                    .and_then(|events| events.into_iter().last())
                    .or_else(|| burst.events.last().cloned());
                Some(MessageRef {
                    provider_msg_id: latest_provider_msg_id.clone(),
                    chat_jid: stored_ref
                        .as_ref()
                        .filter(|r| !r.chat_jid.is_empty())
                        .map(|r| r.chat_jid.clone())
                        .unwrap_or_else(|| chat_jid.clone()),
                    from_me: stored_ref.as_ref().map(|r| r.from_me).unwrap_or(false),
                    text: last_event.as_ref().and_then(|e| e.text()).map(str::to_string),
                })
            };

            let authored = self.secrets.redact(&reply_text);
            let outgoing = self.secrets.expand(&authored);
            let outbound_msg_id = self
                .transport
                .send_text(&chat_jid, &outgoing, reply_target.as_ref())
                .await?;

            let last_event_id = self
                .history_db
                .list_turn_events(&burst.turn_id)
                .ok()
                .and_then(|events| events.into_iter().last())
                .map(|e| e.id)
                .or_else(|| burst.events.last().map(|e| e.id.clone()));

            record_assistant_message(
                &self.history_db,
                &chat_jid,
                &outbound_msg_id,
                &authored,
                Some(burst.turn_id.clone()),
                last_event_id,
            )?;

            info!("Sent reply to {}: {}", chat_jid, authored);
            Ok(())
        }
        .await;

        self.session.set_turn_for("foreground", None);
        result
    }

    /// Tell the owner how the pairing they were asked for turned out.
    ///
    /// One at a time: the code is cached until it resolves, so repeated messages
    /// while pairing is open re-send that same code rather than opening a second
    /// watcher for it.
    fn spawn_login_notifier(&self, chat_jid: String, mut completions: broadcast::Receiver<bool>) {
        use std::sync::atomic::Ordering;
        if self.login_notifier_active.swap(true, Ordering::SeqCst) {
            return;
        }

        let notifier_flag = self.login_notifier_active.clone();
        let transport = self.transport.clone();
        let config = self.config.clone();
        let history_db = self.history_db.clone();

        tokio::spawn(async move {
            if let Ok(success) = completions.recv().await {
                if success {
                    let prompt = format!(
                        "✨ Paired with Codex. What would you like help with, {}?",
                        config.owner_name
                    );
                    match transport.send_text(&chat_jid, &prompt, None).await {
                        Ok(outbound_msg_id) => {
                            if let Err(e) = record_assistant_message(
                                &history_db,
                                &chat_jid,
                                &outbound_msg_id,
                                &prompt,
                                None,
                                None,
                            ) {
                                warn!("Could not record post-pairing greeting: {e:?}");
                            }
                        }
                        Err(e) => {
                            error!("Failed to send paired notification: {e:?}");
                        }
                    }
                }
            }
            notifier_flag.store(false, Ordering::SeqCst);
        });
    }
}

#[cfg(test)]
mod presence_tests {
    use super::*;

    fn event() -> ConversationEvent {
        ConversationEvent::message(
            "m1",
            1,
            "user",
            Some("one more thing".into()),
            None,
            Some("turn1".into()),
            vec![],
        )
    }

    #[test]
    fn test_composing_cannot_hold_a_burst_past_its_normal_ceiling() {
        let mut state = ConversationState::default();
        let mut burst = MessageBurst::new("turn1".into(), event());
        burst.created_at = Instant::now() - MAX_BURST_WAIT;
        burst.last_updated_at = burst.created_at;
        state.bursts.insert("owner:26@lid".into(), burst);

        state.update_presence("owner@s.whatsapp.net", InboundPresenceKind::RecordingAudio);
        assert_eq!(state.remaining_wait("owner:26@lid"), Some(Duration::ZERO));
    }

    #[test]
    fn test_paused_does_not_move_the_burst_deadline() {
        let mut state = ConversationState::default();
        let mut burst = MessageBurst::new("turn1".into(), event());
        burst.created_at = Instant::now() - MAX_BURST_WAIT;
        burst.last_updated_at = burst.created_at;
        state.bursts.insert("owner".into(), burst);
        state.update_presence("owner", InboundPresenceKind::Typing);

        state.update_presence("owner", InboundPresenceKind::Paused);
        let remaining = state.remaining_wait("owner").unwrap();
        assert_eq!(remaining, Duration::ZERO);
    }
}
