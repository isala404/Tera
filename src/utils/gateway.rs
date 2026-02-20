use crate::utils::router::EventRouter;
use crate::utils::store::PostgresStore;
use forge::prelude::*;
use serde_json::Value;
use sqlx::PgPool;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use wacore::download::MediaType;
use wacore_binary::jid::Jid;
use waproto::whatsapp as wa;
use whatsapp_rust::bot::Bot;
use whatsapp_rust::client::Client;
use whatsapp_rust_tokio_transport::TokioWebSocketTransportFactory;
use whatsapp_rust_ureq_http_client::UreqHttpClient;

pub const DOWNLOAD_DIR: &str = "downloads";

const RESTART_DELAY_SECS: u64 = 5;

static ASSET_CACHE: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();

fn asset_cache() -> &'static Mutex<HashMap<String, Vec<u8>>> {
    ASSET_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

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

pub async fn send_asset_media_message(
    chat_id: &str,
    asset_filename: &str,
    media_type_str: &str,
) -> Result<String> {
    tracing::info!(%chat_id, %asset_filename, %media_type_str, "Sending asset media message");
    let client = get_active_client().await?;
    let jid = parse_chat_jid(chat_id)?;

    let media_type = parse_media_type(media_type_str)?;
    let data = resolve_asset(asset_filename, &media_type).await?;
    tracing::info!(size_bytes = data.len(), "Uploading media to WhatsApp");

    let uploaded = client
        .upload(data, media_type)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to upload media: {}", e)))?;
    tracing::info!("Media uploaded, sending message");

    let message = build_media_message(&uploaded, media_type);

    let msg_id = client
        .send_message(jid, message)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to send media message: {}", e)))?;
    tracing::info!(%chat_id, %msg_id, "Media message sent");
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

fn sample_asset_url(filename: &str) -> Option<&'static str> {
    match filename {
        "sample.jpg" => Some("https://picsum.photos/640/480"),
        "sample.mp4" => Some("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"),
        "sample.ogg" => Some("https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_100KB_OGG.ogg"),
        _ => None,
    }
}

async fn resolve_asset(filename: &str, media_type: &MediaType) -> Result<Vec<u8>> {
    // Check in-memory cache first
    {
        let cache = asset_cache().lock().await;
        if let Some(data) = cache.get(filename) {
            tracing::info!(%filename, "Using cached asset");
            return Ok(data.clone());
        }
    }

    let url = sample_asset_url(filename).ok_or_else(|| {
        ForgeError::NotFound(format!("Unknown sample asset: {}", filename))
    })?;

    tracing::info!(%url, %filename, "Downloading sample asset");
    let response = reqwest::get(url)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to download asset: {}", e)))?;

    if !response.status().is_success() {
        return Err(ForgeError::Internal(format!(
            "Asset download returned {}",
            response.status()
        )));
    }

    let raw = response
        .bytes()
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to read asset response: {}", e)))?
        .to_vec();
    tracing::info!(size_bytes = raw.len(), "Downloaded asset");

    let data = if matches!(media_type, MediaType::Audio) {
        convert_audio_in_memory(&raw)?
    } else {
        raw
    };

    // Cache the final (possibly converted) bytes
    asset_cache().lock().await.insert(filename.to_string(), data.clone());
    Ok(data)
}

fn convert_audio_in_memory(raw: &[u8]) -> Result<Vec<u8>> {
    let tmp_in = std::env::temp_dir().join("tera_audio_in.ogg");
    let tmp_out = std::env::temp_dir().join("tera_audio_out.ogg");

    fs::write(&tmp_in, raw)
        .map_err(|e| ForgeError::Internal(format!("Failed to write temp audio: {}", e)))?;

    let converted_path = convert_audio_for_whatsapp(&tmp_in).map_err(ForgeError::Internal)?;
    let data = fs::read(&converted_path)
        .map_err(|e| ForgeError::Internal(format!("Failed to read converted audio: {}", e)))?;

    // Clean up temp files
    let _ = fs::remove_file(&tmp_in);
    let _ = fs::remove_file(&tmp_out);
    let _ = fs::remove_file(&converted_path);

    Ok(data)
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

pub fn convert_audio_for_whatsapp(input_path: &Path) -> std::result::Result<PathBuf, String> {
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid input path")?;
    let output_path = std::env::temp_dir().join(format!("{}_converted.ogg", stem));

    tracing::info!(
        "Converting audio to WhatsApp voice note format: {:?}",
        output_path
    );

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
        .map_err(|e| format!("Failed to execute ffmpeg: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(output_path)
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
    if let Err(err) = fs::create_dir_all(DOWNLOAD_DIR) {
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
