use crate::utils::router::EventRouter;
use crate::utils::store::PostgresStore;
use forge::prelude::*;
use serde_json::Value;
use sqlx::PgPool;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::RwLock;
use wacore::download::MediaType;
use wacore_binary::jid::Jid;
use waproto::whatsapp as wa;
use whatsapp_rust::bot::Bot;
use whatsapp_rust::client::Client;
use whatsapp_rust_tokio_transport::TokioWebSocketTransportFactory;
use whatsapp_rust_ureq_http_client::UreqHttpClient;

pub const DOWNLOAD_DIR: &str = "downloads";

const RESTART_DELAY_SECS: u64 = 5;

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
    tracing::info!(%chat_id, text_len = text.len(), "Sending text message");
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;
    let message = wa::Message {
        conversation: Some(text.to_string()),
        ..Default::default()
    };

    let msg_id = client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send text message: {}", e)))?;
    tracing::info!(%chat_id, %msg_id, "Text message sent");
    Ok(msg_id)
}

pub async fn send_reaction_message(
    chat_id: &str,
    target_message_id: &str,
    participant: Option<&str>,
    emoji: &str,
) -> Result<String> {
    tracing::info!(%chat_id, %target_message_id, %emoji, "Sending reaction");
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

pub async fn send_file_message(
    chat_id: &str,
    file_path: &str,
    media_type_str: &str,
) -> Result<String> {
    tracing::info!(%chat_id, %file_path, %media_type_str, "Sending file message");
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;

    let media_type = parse_media_type(media_type_str)?;
    let data = tokio::fs::read(file_path).await.map_err(|e| {
        ForgeError::Internal(format!("Failed to read file {}: {}", file_path, e))
    })?;

    // audio needs ogg opus conversion for WhatsApp voice notes
    let data = if matches!(media_type, MediaType::Audio) {
        convert_to_voice_note(file_path).unwrap_or(data)
    } else {
        data
    };

    tracing::info!(size_bytes = data.len(), "Uploading file to WhatsApp");

    let uploaded = client
        .upload(data, media_type)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to upload media: {}", e)))?;

    let mimetype = mime_from_path(file_path, media_type_str);
    let message = build_media_message(&uploaded, media_type, &mimetype);

    let msg_id = client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send media message: {}", e)))?;
    tracing::info!(%chat_id, %msg_id, "File message sent");
    Ok(msg_id)
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

pub fn mime_from_path(file_path: &str, media_type: &str) -> String {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match (media_type, ext.to_lowercase().as_str()) {
        ("image", "png") => "image/png",
        ("image", "webp") => "image/webp",
        ("image", "gif") => "image/gif",
        ("image", _) => "image/jpeg",
        ("video", _) => "video/mp4",
        ("audio", "mp3") => "audio/mpeg",
        ("audio", "wav") => "audio/wav",
        ("audio", _) => "audio/ogg; codecs=opus",
        _ => "application/octet-stream",
    }
    .to_string()
}

fn convert_to_voice_note(input_path: &str) -> Option<Vec<u8>> {
    let output_path = std::env::temp_dir().join("tera_voice_converted.ogg");

    let output = Command::new("ffmpeg")
        .arg("-i")
        .arg(input_path)
        .arg("-c:a")
        .arg("libopus")
        .arg("-b:a")
        .arg("16k")
        .arg("-vbr")
        .arg("on")
        .arg("-ac")
        .arg("1")
        .arg("-application")
        .arg("voip")
        .arg("-map_metadata")
        .arg("-1")
        .arg(&output_path)
        .arg("-y")
        .output()
        .ok()?;

    if !output.status.success() {
        tracing::warn!(
            "ffmpeg conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let data = std::fs::read(&output_path).ok()?;
    let _ = std::fs::remove_file(&output_path);
    Some(data)
}

fn build_media_message(
    uploaded: &whatsapp_rust::upload::UploadResponse,
    media_type: MediaType,
    mimetype: &str,
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
                mimetype: Some(mimetype.to_string()),
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
                mimetype: Some(mimetype.to_string()),
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
                mimetype: Some(mimetype.to_string()),
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

pub fn spawn_whatsapp_gateway(pool: Arc<PgPool>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        run_whatsapp_gateway_forever(pool).await;
    })
}

async fn run_whatsapp_gateway_forever(pool: Arc<PgPool>) {
    loop {
        if let Err(err) = run_gateway_once(pool.clone()).await {
            tracing::error!("WhatsApp gateway error: {}", err);
        }
        tokio::time::sleep(Duration::from_secs(RESTART_DELAY_SECS)).await;
    }
}

async fn run_gateway_once(pool: Arc<PgPool>) -> Result<()> {
    if let Err(err) = std::fs::create_dir_all(DOWNLOAD_DIR) {
        tracing::error!("Failed to create downloads directory: {}", err);
    }
    let backend = Arc::new(PostgresStore::new(pool.clone()));
    let router = EventRouter::new((*pool).clone());
    let router_for_bot = router.clone();

    let mut bot = Bot::builder()
        .with_backend(backend)
        .with_transport_factory(TokioWebSocketTransportFactory::new())
        .with_http_client(UreqHttpClient::new())
        .skip_history_sync()
        .on_event(move |event, client| {
            let router = router_for_bot.clone();
            async move {
                router.handle_event(event, client).await;
            }
        })
        .build()
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to build WhatsApp bot: {}", e)))?;

    let runner = bot
        .run()
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to start WhatsApp bot: {}", e)))?;

    tracing::info!("WhatsApp gateway running");
    runner.await.map_err(|e| {
        ForgeError::Internal(format!("WhatsApp runner task failed unexpectedly: {}", e))
    })?;

    tracing::warn!("WhatsApp runner exited");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{media_json_kind, mime_from_path, parse_media_type};
    use wacore::download::MediaType;

    #[test]
    fn test_parse_media_type() {
        assert!(matches!(parse_media_type("image").unwrap(), MediaType::Image));
        assert!(matches!(parse_media_type("video").unwrap(), MediaType::Video));
        assert!(matches!(parse_media_type("audio").unwrap(), MediaType::Audio));
        assert!(parse_media_type("document").is_err());
    }

    #[test]
    fn test_media_json_kind() {
        let value = serde_json::json!({"kind": "file"});
        assert_eq!(media_json_kind(&value), Some("file"));
    }

    #[test]
    fn test_mime_from_path() {
        assert_eq!(mime_from_path("/tmp/photo.png", "image"), "image/png");
        assert_eq!(mime_from_path("/tmp/photo.jpg", "image"), "image/jpeg");
        assert_eq!(mime_from_path("/tmp/video.mp4", "video"), "video/mp4");
        assert_eq!(mime_from_path("/tmp/voice.ogg", "audio"), "audio/ogg; codecs=opus");
        assert_eq!(mime_from_path("/tmp/voice.mp3", "audio"), "audio/mpeg");
    }
}
