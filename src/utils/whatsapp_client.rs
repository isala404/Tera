#![allow(dead_code)]

use crate::utils::whatsapp_helpers::{ASSETS_DIR, convert_audio_for_whatsapp};
use forge::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use wacore::download::MediaType;
use wacore_binary::jid::Jid;
use waproto::whatsapp as wa;
use whatsapp_rust::client::Client;

static ACTIVE_CLIENT: OnceLock<RwLock<Option<Arc<Client>>>> = OnceLock::new();

fn client_slot() -> &'static RwLock<Option<Arc<Client>>> {
    ACTIVE_CLIENT.get_or_init(|| RwLock::new(None))
}

pub async fn set_active_client(client: Arc<Client>) {
    *client_slot().write().await = Some(client);
}

async fn get_active_client() -> Result<Arc<Client>> {
    client_slot()
        .read()
        .await
        .as_ref()
        .cloned()
        .ok_or_else(|| ForgeError::Internal("WhatsApp client is not connected".to_string()))
}

fn parse_chat_jid(chat_id: &str) -> Result<Jid> {
    Jid::from_str(chat_id).map_err(|e| ForgeError::Validation(format!("Invalid chat JID: {}", e)))
}

pub async fn send_composing(chat_id: &str) -> Result<()> {
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;
    client
        .chatstate()
        .send_composing(&jid)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send composing state: {}", e)))
}

pub async fn send_paused(chat_id: &str) -> Result<()> {
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;
    client
        .chatstate()
        .send_paused(&jid)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send paused state: {}", e)))
}

pub async fn send_text_message(chat_id: &str, text: &str) -> Result<String> {
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;
    let message = wa::Message {
        conversation: Some(text.to_string()),
        ..Default::default()
    };

    client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send text message: {}", e)))
}

pub async fn send_reaction_message(
    chat_id: &str,
    target_message_id: &str,
    participant: Option<&str>,
    emoji: &str,
) -> Result<String> {
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;

    let key = wa::MessageKey {
        remote_jid: Some(chat_id.to_string()),
        id: Some(target_message_id.to_string()),
        from_me: Some(false),
        participant: participant.map(ToOwned::to_owned),
    };

    let message = wa::Message {
        reaction_message: Some(wa::message::ReactionMessage {
            key: Some(key),
            text: Some(emoji.to_string()),
            sender_timestamp_ms: Some(chrono::Utc::now().timestamp_millis()),
            ..Default::default()
        }),
        ..Default::default()
    };

    client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send reaction: {}", e)))
}

pub async fn send_asset_media_message(
    chat_id: &str,
    asset_filename: &str,
    media_type: &str,
) -> Result<String> {
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;

    let media_type = parse_media_type(media_type)?;
    let path = prepare_media_path(asset_filename, &media_type)?;
    let data = fs::read(&path)
        .map_err(|e| ForgeError::Internal(format!("Failed to read media asset: {}", e)))?;

    let uploaded = client
        .upload(data, media_type)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to upload media: {}", e)))?;

    let message = build_media_message(&uploaded, media_type);

    client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send media message: {}", e)))
}

fn parse_media_type(media_type: &str) -> Result<MediaType> {
    match media_type {
        "image" => Ok(MediaType::Image),
        "video" => Ok(MediaType::Video),
        "audio" => Ok(MediaType::Audio),
        _ => Err(ForgeError::Validation(format!(
            "Unsupported outbound media type: {}",
            media_type
        ))),
    }
}

fn prepare_media_path(filename: &str, media_type: &MediaType) -> Result<PathBuf> {
    let path = PathBuf::from(ASSETS_DIR).join(filename);
    if !path.exists() {
        return Err(ForgeError::NotFound(format!(
            "Media asset not found: {}",
            path.display()
        )));
    }

    if matches!(media_type, MediaType::Audio) {
        convert_audio_for_whatsapp(&path).map_err(ForgeError::Internal)
    } else {
        Ok(path)
    }
}

fn build_media_message(
    uploaded: &whatsapp_rust::upload::UploadResponse,
    media_type: MediaType,
) -> wa::Message {
    match media_type {
        MediaType::Image => wa::Message {
            image_message: Some(Box::new(wa::message::ImageMessage {
                url: Some(uploaded.url.clone()),
                direct_path: Some(uploaded.direct_path.clone()),
                media_key: Some(uploaded.media_key.clone()),
                file_enc_sha256: Some(uploaded.file_enc_sha256.clone()),
                file_sha256: Some(uploaded.file_sha256.clone()),
                file_length: Some(uploaded.file_length),
                mimetype: Some("image/jpeg".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        },
        MediaType::Video => wa::Message {
            video_message: Some(Box::new(wa::message::VideoMessage {
                url: Some(uploaded.url.clone()),
                direct_path: Some(uploaded.direct_path.clone()),
                media_key: Some(uploaded.media_key.clone()),
                file_enc_sha256: Some(uploaded.file_enc_sha256.clone()),
                file_sha256: Some(uploaded.file_sha256.clone()),
                file_length: Some(uploaded.file_length),
                mimetype: Some("video/mp4".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        },
        MediaType::Audio => wa::Message {
            audio_message: Some(Box::new(wa::message::AudioMessage {
                url: Some(uploaded.url.clone()),
                direct_path: Some(uploaded.direct_path.clone()),
                media_key: Some(uploaded.media_key.clone()),
                file_enc_sha256: Some(uploaded.file_enc_sha256.clone()),
                file_sha256: Some(uploaded.file_sha256.clone()),
                file_length: Some(uploaded.file_length),
                mimetype: Some("audio/ogg; codecs=opus".to_string()),
                ptt: Some(true),
                ..Default::default()
            })),
            ..Default::default()
        },
        _ => wa::Message::default(),
    }
}

pub fn media_json_kind(media: &Value) -> Option<&str> {
    media.get("kind").and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::{media_json_kind, parse_media_type};
    use wacore::download::MediaType;

    #[test]
    fn test_parse_media_type() {
        assert!(matches!(
            parse_media_type("image").unwrap(),
            MediaType::Image
        ));
        assert!(matches!(
            parse_media_type("video").unwrap(),
            MediaType::Video
        ));
        assert!(matches!(
            parse_media_type("audio").unwrap(),
            MediaType::Audio
        ));
        assert!(parse_media_type("document").is_err());
    }

    #[test]
    fn test_media_json_kind() {
        let value = serde_json::json!({"kind": "asset_media"});
        assert_eq!(media_json_kind(&value), Some("asset_media"));
    }
}
