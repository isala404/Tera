#![allow(dead_code)]

use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use wacore::proto_helpers::MessageExt;
use wacore::types::events::Event;
use wacore::types::message::MessageInfo;
use wacore::types::presence::ChatPresence;
use waproto::whatsapp as wa;
use whatsapp_rust::bot::MessageContext;
use whatsapp_rust::client::Client;

use crate::utils::whatsapp_buffer::{BufferManager, TypingState};
use crate::utils::whatsapp_handlers::{
    AuthResult, check_user_auth, create_new_user, handle_expired_code, handle_new_user,
    handle_verification_attempt, process_with_cancellation, regenerate_pairing_code,
    replay_cached_response, update_last_seen,
};
use crate::utils::whatsapp_helpers::send_reaction_to_message;
use crate::utils::whatsapp_task::{MessageKey, TaskManager, TaskStateSnapshot};

const FLUSH_CHECK_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct EventRouter {
    task_manager: TaskManager,
    buffer_manager: BufferManager,
    user_db: PgPool,
}

impl EventRouter {
    pub fn new(user_db: PgPool) -> Self {
        Self {
            task_manager: TaskManager::new(),
            buffer_manager: BufferManager::new(),
            user_db,
        }
    }

    pub fn task_manager(&self) -> &TaskManager {
        &self.task_manager
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
                self.handle_presence_event(&chat_jid, typing_state).await;
            }
            Event::Connected(_) => {
                tracing::info!("Hospital Support Bot Connected!");
            }
            Event::PairingQrCode { code, .. } => {
                tracing::info!("Scan this QR code to pair:");
                qr2term::print_qr(code).expect("Failed to print QR code");
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

        if let Some(inner) = msg.edited_message.as_ref().and_then(|e| e.message.as_ref()) {
            tracing::debug!(
                "DEBUG EDIT inner: conv={:?}, ext_text={}, protocol={}, edit_msg={}",
                inner.conversation,
                inner.extended_text_message.is_some(),
                inner.protocol_message.is_some(),
                inner.edited_message.is_some()
            );

            if let Some(ref pm) = inner.protocol_message {
                tracing::debug!(
                    "DEBUG EDIT inner.protocol_message: type={:?}, key_id={:?}",
                    pm.r#type,
                    pm.key.as_ref().and_then(|k| k.id.as_ref())
                );

                if let Some(original_id) = pm.key.as_ref().and_then(|k| k.id.as_ref()) {
                    let edited_content = pm
                        .edited_message
                        .as_ref()
                        .map(|m| (**m).clone())
                        .unwrap_or_default();

                    tracing::info!(
                        "Edit detected for message {} (via inner.protocol_message)",
                        original_id
                    );
                    self.handle_edit(chat_jid, original_id.clone(), edited_content, info, client)
                        .await;
                    return;
                }
            }
        }

        self.handle_new_message(chat_jid, sender_jid, msg, info, client)
            .await;
    }

    async fn handle_edit(
        &self,
        chat_jid: String,
        original_id: String,
        new_msg: wa::Message,
        info: MessageInfo,
        client: Arc<Client>,
    ) {
        let key = MessageKey::new(chat_jid.clone(), original_id.clone());

        if self
            .buffer_manager
            .update_buffered_message(&chat_jid, &original_id, new_msg.clone())
            .await
        {
            tracing::info!("Updated buffered message {}", original_id);
            return;
        }

        match self.task_manager.get(&key).await {
            Some(TaskStateSnapshot::InProgress) => {
                tracing::info!(
                    "Cancelling in-progress task for edited message {}",
                    original_id
                );
                self.task_manager.cancel(&key).await;

                let ctx = MessageContext {
                    message: Box::new(new_msg),
                    info,
                    client,
                };
                self.spawn_processing_task(key, ctx).await;
            }
            Some(TaskStateSnapshot::Completed { result }) => {
                tracing::info!(
                    "Edit received for completed message {}, sending warning",
                    original_id
                );
                let ctx = MessageContext {
                    message: Box::new(new_msg),
                    info,
                    client,
                };

                send_reaction_to_message(&ctx, &original_id, "\u{26a0}\u{fe0f}").await;
                replay_cached_response(&ctx, &result).await;
            }
            Some(TaskStateSnapshot::Cancelled) | None => {
                tracing::info!(
                    "Edit received for unknown/cancelled message {}, ignoring",
                    original_id
                );
            }
        }
    }

    async fn handle_new_message(
        &self,
        chat_jid: String,
        sender_jid: String,
        msg: wa::Message,
        info: MessageInfo,
        client: Arc<Client>,
    ) {
        let ctx = MessageContext {
            message: Box::new(msg.clone()),
            info: info.clone(),
            client: client.clone(),
        };

        match check_user_auth(&self.user_db, &sender_jid).await {
            AuthResult::NewUser => {
                create_new_user(&self.user_db, &sender_jid).await;
                handle_new_user(&ctx).await;
                return;
            }
            AuthResult::PendingVerification { expired } => {
                if expired {
                    regenerate_pairing_code(&self.user_db, &sender_jid).await;
                    handle_expired_code(&ctx).await;
                    return;
                }

                if let Some(text) = ctx.message.text_content() {
                    handle_verification_attempt(&ctx, &self.user_db, &sender_jid, text.trim())
                        .await;
                }
                return;
            }
            AuthResult::Authenticated => {
                update_last_seen(&self.user_db, &sender_jid).await;
            }
        }

        self.buffer_manager.add(chat_jid.clone(), msg, info).await;
        self.schedule_flush_check(chat_jid, client).await;
    }

    async fn handle_presence_event(&self, chat_jid: &str, state: TypingState) {
        self.buffer_manager.update_typing(chat_jid, state).await;
    }

    async fn schedule_flush_check(&self, chat_jid: String, client: Arc<Client>) {
        let buffer_manager = self.buffer_manager.clone();
        let task_manager = self.task_manager.clone();

        tokio::spawn(async move {
            loop {
                sleep(FLUSH_CHECK_INTERVAL).await;

                if !buffer_manager.has_pending(&chat_jid).await {
                    break;
                }

                let ready_chats = buffer_manager.get_ready_chats().await;
                if ready_chats.contains(&chat_jid) {
                    let messages = buffer_manager.flush(&chat_jid).await;
                    for buffered in messages {
                        let key = MessageKey::new(chat_jid.clone(), buffered.info.id.clone());
                        let ctx = MessageContext {
                            message: Box::new(buffered.message),
                            info: buffered.info,
                            client: client.clone(),
                        };

                        spawn_task(task_manager.clone(), key, ctx);
                    }
                    break;
                }
            }
        });
    }

    async fn spawn_processing_task(&self, key: MessageKey, ctx: MessageContext) {
        spawn_task(self.task_manager.clone(), key, ctx);
    }
}

fn spawn_task(task_manager: TaskManager, key: MessageKey, ctx: MessageContext) {
    tokio::spawn(async move {
        let cancel_token = task_manager.start(key.clone()).await;
        let message_id = key.message_id.clone();

        if let Some(result) = process_with_cancellation(ctx, cancel_token).await {
            task_manager.complete(key, result).await;
        } else {
            tracing::info!("Task for message {} was cancelled", message_id);
        }
    });
}
