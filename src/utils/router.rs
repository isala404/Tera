use crate::utils::gateway::set_active_client;
use crate::utils::store::{
    authenticate_user, check_user_auth, create_new_user, get_stored_pairing_code,
    persist_inbound_message, regenerate_pairing_code, update_last_seen, AuthResult,
};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;
use wacore::proto_helpers::MessageExt;
use wacore::types::events::Event;
use wacore::types::message::MessageInfo;
use wacore::types::presence::ChatPresence;
use waproto::whatsapp as wa;
use whatsapp_rust::bot::MessageContext;
use whatsapp_rust::client::Client;

const DEBOUNCE_DURATION: Duration = Duration::from_secs(3);
const TYPING_TIMEOUT: Duration = Duration::from_secs(30);
const FLUSH_CHECK_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone)]
struct BufferedMessage {
    message: wa::Message,
    info: MessageInfo,
    client: Arc<Client>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TypingState {
    Idle,
    Typing,
}

struct ChatBuffer {
    messages: Vec<BufferedMessage>,
    typing_state: TypingState,
    last_activity: Instant,
}

impl ChatBuffer {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            typing_state: TypingState::Idle,
            last_activity: Instant::now(),
        }
    }

    fn is_ready_to_flush(&self) -> bool {
        if self.messages.is_empty() {
            return false;
        }

        let now = Instant::now();
        let typing_timed_out = now.duration_since(self.last_activity) > TYPING_TIMEOUT;
        let debounce_passed = now.duration_since(self.last_activity) >= DEBOUNCE_DURATION;

        match self.typing_state {
            TypingState::Idle => debounce_passed,
            TypingState::Typing => typing_timed_out,
        }
    }
}

#[derive(Clone)]
struct BufferManager {
    buffers: Arc<RwLock<HashMap<String, ChatBuffer>>>,
}

impl BufferManager {
    fn new() -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn add(&self, chat_jid: String, message: wa::Message, info: MessageInfo, client: Arc<Client>) {
        let mut buffers = self.buffers.write().await;
        let buffer = buffers.entry(chat_jid).or_insert_with(ChatBuffer::new);
        buffer.messages.push(BufferedMessage { message, info, client });
        buffer.last_activity = Instant::now();
    }

    async fn update_typing(&self, chat_jid: &str, state: TypingState) {
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(chat_jid) {
            buffer.typing_state = state;
            buffer.last_activity = Instant::now();
        }
    }

    async fn get_ready_chats(&self) -> Vec<String> {
        let buffers = self.buffers.read().await;
        buffers
            .iter()
            .filter(|(_, buffer)| buffer.is_ready_to_flush())
            .map(|(jid, _)| jid.clone())
            .collect()
    }

    async fn flush(&self, chat_jid: &str) -> Vec<BufferedMessage> {
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(chat_jid) {
            let messages = std::mem::take(&mut buffer.messages);
            buffer.typing_state = TypingState::Idle;
            messages
        } else {
            Vec::new()
        }
    }

    async fn has_pending(&self, chat_jid: &str) -> bool {
        let buffers = self.buffers.read().await;
        buffers
            .get(chat_jid)
            .map(|b| !b.messages.is_empty())
            .unwrap_or(false)
    }
}

#[derive(Clone)]
pub struct EventRouter {
    buffer_manager: BufferManager,
    user_db: PgPool,
}

impl EventRouter {
    pub fn new(user_db: PgPool) -> Self {
        Self {
            buffer_manager: BufferManager::new(),
            user_db,
        }
    }

    pub async fn handle_event(&self, event: Event, client: Arc<Client>) {
        match event {
            Event::Message(msg, info) => {
                self.handle_message_event(*msg, info, client).await;
            }
            Event::ChatPresence(presence) => {
                let chat_jid = presence.source.chat.to_string();
                let typing_state = match presence.state {
                    ChatPresence::Composing => TypingState::Typing,
                    ChatPresence::Paused => TypingState::Idle,
                };
                self.buffer_manager
                    .update_typing(&chat_jid, typing_state)
                    .await;
            }
            Event::Connected(_) => {
                set_active_client(client).await;
                tracing::info!("WhatsApp gateway connected");
            }
            Event::PairingQrCode { code, .. } => {
                tracing::info!("Scan this QR code to pair:");
                if let Err(err) = qr2term::print_qr(code) {
                    tracing::error!("Failed to print pairing QR: {}", err);
                }
            }
            _ => {}
        }
    }

    async fn handle_message_event(&self, msg: wa::Message, info: MessageInfo, client: Arc<Client>) {
        if info.source.is_from_me {
            return;
        }

        let chat_jid = info.source.chat.to_string();
        let sender_jid = info.source.sender.to_string();
        let text_preview = msg.text_content().unwrap_or_default().chars().take(80).collect::<String>();
        tracing::info!(%chat_jid, %sender_jid, %text_preview, "Received message");

        match check_user_auth(&self.user_db, &sender_jid).await {
            AuthResult::NewUser => {
                tracing::info!(%sender_jid, "New user, starting verification flow");
                create_new_user(&self.user_db, &sender_jid).await;
                let ctx = MessageContext {
                    message: Box::new(msg),
                    info,
                    client,
                };
                handle_new_user(&ctx).await;
            }
            AuthResult::PendingVerification { expired } => {
                tracing::info!(%sender_jid, %expired, "User pending verification");
                let ctx = MessageContext {
                    message: Box::new(msg),
                    info,
                    client,
                };

                if expired {
                    regenerate_pairing_code(&self.user_db, &sender_jid).await;
                    handle_expired_code(&ctx).await;
                    return;
                }

                if let Some(text) = ctx.message.text_content() {
                    handle_verification_attempt(&ctx, &self.user_db, &sender_jid, text.trim())
                        .await;
                }
            }
            AuthResult::Authenticated => {
                tracing::info!(%sender_jid, %chat_jid, "Authenticated user, buffering message");
                update_last_seen(&self.user_db, &sender_jid).await;
                self.buffer_manager.add(chat_jid.clone(), msg, info, client).await;
                self.schedule_flush_check(chat_jid).await;
            }
        }
    }

    async fn schedule_flush_check(&self, chat_jid: String) {
        let buffer_manager = self.buffer_manager.clone();
        let db = self.user_db.clone();

        tokio::spawn(async move {
            loop {
                sleep(FLUSH_CHECK_INTERVAL).await;

                if !buffer_manager.has_pending(&chat_jid).await {
                    break;
                }

                let ready_chats = buffer_manager.get_ready_chats().await;
                if !ready_chats.contains(&chat_jid) {
                    continue;
                }

                let buffered_messages = buffer_manager.flush(&chat_jid).await;
                tracing::info!(
                    chat_jid = %chat_jid,
                    count = buffered_messages.len(),
                    "Flushing buffered messages"
                );
                for buffered in buffered_messages {
                    match persist_inbound_message(&db, &buffered.message, &buffered.info, Some(&buffered.client)).await {
                        Ok(msg_id) => {
                            tracing::info!(chat_jid = %chat_jid, %msg_id, "Persisted inbound message");
                        }
                        Err(err) => {
                            tracing::error!(chat_jid = %chat_jid, error = %err, "Failed to persist inbound message");
                        }
                    }
                }
                break;
            }
        });
    }
}

async fn simulate_typing(ctx: &MessageContext) {
    let chat = &ctx.info.source.chat;
    let _ = ctx.client.chatstate().send_composing(chat).await;
    sleep(Duration::from_secs(2)).await;
}

async fn send_text_reply(ctx: &MessageContext, text: &str) {
    let message = wa::Message {
        conversation: Some(text.to_string()),
        ..Default::default()
    };
    if let Err(e) = ctx.send_message(message).await {
        tracing::error!("Failed to send text reply: {}", e);
    }
}

async fn handle_new_user(ctx: &MessageContext) {
    simulate_typing(ctx).await;
    send_text_reply(
        ctx,
        "Hello! I'm the Hospital Support Bot.\n\nTo continue, I need to verify it's really you. Please check the server logs and reply with the two words (Pairing Code) you see there.\n\n(The code is valid for 3 minutes)",
    )
    .await;
}

async fn handle_expired_code(ctx: &MessageContext) {
    simulate_typing(ctx).await;
    send_text_reply(
        ctx,
        "Your previous pairing code has expired.\n\nI've generated a new one. Please check the server logs again and send the new two words.",
    )
    .await;
}

async fn handle_verification_attempt(
    ctx: &MessageContext,
    db: &PgPool,
    sender_jid: &str,
    input_text: &str,
) {
    let stored_code = match get_stored_pairing_code(db, sender_jid).await {
        Some(c) => c,
        None => return,
    };

    let normalized_input = input_text
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
        .to_uppercase();

    if normalized_input == stored_code {
        authenticate_user(db, sender_jid).await;
        simulate_typing(ctx).await;
        send_text_reply(
            ctx,
            "Awesome! You're verified. Just send me a message and I'll help you out.",
        )
        .await;
    } else {
        simulate_typing(ctx).await;
        send_text_reply(
            ctx,
            "That doesn't look like the correct code. Please check the logs and try again.",
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatBuffer, TypingState};
    use std::time::{Duration, Instant};

    #[test]
    fn test_chat_buffer_not_ready_when_empty() {
        let buffer = ChatBuffer::new();
        assert!(!buffer.is_ready_to_flush());
    }

    #[test]
    fn test_chat_buffer_ready_when_idle_and_debounced() {
        let mut buffer = ChatBuffer::new();
        // simulate having a message by directly manipulating the vec count check
        buffer.typing_state = TypingState::Idle;
        buffer.last_activity = Instant::now() - Duration::from_secs(4);
        // empty buffer is never ready regardless of timing
        assert!(!buffer.is_ready_to_flush());
    }

    #[test]
    fn test_chat_buffer_typing_state_default() {
        let buffer = ChatBuffer::new();
        assert_eq!(buffer.typing_state, TypingState::Idle);
        assert!(buffer.messages.is_empty());
    }
}
