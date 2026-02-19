#![allow(dead_code)]

use crate::utils::whatsapp_buffer::{BufferManager, TypingState};
use crate::utils::whatsapp_client::set_active_client;
use crate::utils::whatsapp_handlers::{
    AuthResult, check_user_auth, create_new_user, handle_expired_code, handle_new_user,
    handle_verification_attempt, regenerate_pairing_code, update_last_seen,
};
use forge::prelude::*;
use serde_json::{Value, json};
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use wacore::proto_helpers::MessageExt;
use wacore::types::events::Event;
use wacore::types::message::MessageInfo;
use wacore::types::presence::ChatPresence;
use waproto::whatsapp as wa;
use whatsapp_rust::client::Client;

const FLUSH_CHECK_INTERVAL: Duration = Duration::from_millis(500);

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
                self.handle_presence_event(&chat_jid, typing_state).await;
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

        match check_user_auth(&self.user_db, &sender_jid).await {
            AuthResult::NewUser => {
                create_new_user(&self.user_db, &sender_jid).await;
                let ctx = whatsapp_rust::bot::MessageContext {
                    message: Box::new(msg),
                    info,
                    client,
                };
                handle_new_user(&ctx).await;
            }
            AuthResult::PendingVerification { expired } => {
                let ctx = whatsapp_rust::bot::MessageContext {
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
                update_last_seen(&self.user_db, &sender_jid).await;
                self.buffer_manager.add(chat_jid.clone(), msg, info).await;
                self.schedule_flush_check(chat_jid).await;
            }
        }
    }

    async fn handle_presence_event(&self, chat_jid: &str, state: TypingState) {
        self.buffer_manager.update_typing(chat_jid, state).await;
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
                for buffered in buffered_messages {
                    if let Err(err) =
                        persist_inbound_message(&db, &buffered.message, &buffered.info).await
                    {
                        tracing::error!("Failed to persist inbound message: {}", err);
                    }
                }
                break;
            }
        });
    }
}

pub async fn persist_inbound_message(
    db: &PgPool,
    msg: &wa::Message,
    info: &MessageInfo,
) -> Result<Uuid> {
    let media = extract_inbound_media(msg);
    let content_text = if media.is_some() {
        None
    } else {
        msg.text_content().map(ToOwned::to_owned)
    };

    let row_id = Uuid::new_v4();
    let metadata = json!({
        "whatsapp_message_id": info.id,
        "chat_jid": info.source.chat.to_string(),
        "sender_jid": info.source.sender.to_string(),
        "is_group": info.source.is_group,
        "is_from_me": info.source.is_from_me,
        "received_at": chrono::Utc::now(),
    });

    sqlx::query(
        r#"
        INSERT INTO whatsapp_messages (
            id, chat_id, direction, status, content_text, media, metadata
        )
        VALUES (
            $1,
            $2,
            $3::message_direction,
            $4::message_status,
            $5,
            $6,
            $7
        )
        "#,
    )
    .bind(row_id)
    .bind(info.source.chat.to_string())
    .bind("in")
    .bind("pending_agent")
    .bind(content_text)
    .bind(media)
    .bind(metadata)
    .execute(db)
    .await?;

    Ok(row_id)
}

pub fn extract_inbound_media(msg: &wa::Message) -> Option<Value> {
    let base = msg.get_base_message();

    if let Some(image) = &base.image_message {
        return Some(json!({
            "kind": "image",
            "caption": image.caption,
            "mimetype": image.mimetype,
            "file_length": image.file_length,
        }));
    }

    if let Some(video) = &base.video_message {
        return Some(json!({
            "kind": "video",
            "caption": video.caption,
            "mimetype": video.mimetype,
            "file_length": video.file_length,
        }));
    }

    if let Some(audio) = &base.audio_message {
        return Some(json!({
            "kind": "audio",
            "mimetype": audio.mimetype,
            "file_length": audio.file_length,
            "ptt": audio.ptt,
            "seconds": audio.seconds,
        }));
    }

    if let Some(document) = &base.document_message {
        return Some(json!({
            "kind": "document",
            "title": document.title,
            "file_name": document.file_name,
            "mimetype": document.mimetype,
            "file_length": document.file_length,
        }));
    }

    if let Some(sticker) = &base.sticker_message {
        return Some(json!({
            "kind": "sticker",
            "mimetype": sticker.mimetype,
            "is_animated": sticker.is_animated,
        }));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::extract_inbound_media;
    use waproto::whatsapp as wa;

    #[test]
    fn test_extract_inbound_media_image() {
        let msg = wa::Message {
            image_message: Some(Box::new(wa::message::ImageMessage {
                caption: Some("caption".to_string()),
                mimetype: Some("image/jpeg".to_string()),
                file_length: Some(123),
                ..Default::default()
            })),
            ..Default::default()
        };

        let media = extract_inbound_media(&msg).expect("image media should be extracted");
        assert_eq!(media["kind"], "image");
        assert_eq!(media["caption"], "caption");
    }

    #[test]
    fn test_extract_inbound_media_none_for_plain_text() {
        let msg = wa::Message {
            conversation: Some("hello".to_string()),
            ..Default::default()
        };

        assert!(extract_inbound_media(&msg).is_none());
    }
}
