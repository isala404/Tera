#![allow(dead_code)]

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tokio::time::sleep;
use wacore::download::{Downloadable, MediaType};
use waproto::whatsapp as wa;
use whatsapp_rust::bot::MessageContext;

pub const DOWNLOAD_DIR: &str = "downloads";
pub const ASSETS_DIR: &str = "assets";

pub async fn simulate_typing(ctx: &MessageContext) {
    let chat = &ctx.info.source.chat;
    let _ = ctx.client.chatstate().send_composing(chat).await;
    sleep(Duration::from_secs(2)).await;
}

pub async fn simulate_recording(ctx: &MessageContext) {
    let chat = &ctx.info.source.chat;
    let _ = ctx.client.chatstate().send_recording(chat).await;
    sleep(Duration::from_secs(2)).await;
}

pub async fn send_text_reply(ctx: &MessageContext, text: &str) {
    let message = wa::Message {
        conversation: Some(text.to_string()),
        ..Default::default()
    };
    if let Err(e) = ctx.send_message(message).await {
        tracing::error!("Failed to send text reply: {}", e);
    }
}

pub async fn send_reaction(ctx: &MessageContext, emoji: &str) {
    let key = wa::MessageKey {
        remote_jid: Some(ctx.info.source.chat.to_string()),
        id: Some(ctx.info.id.clone()),
        from_me: Some(ctx.info.source.is_from_me),
        participant: if ctx.info.source.is_group {
            Some(ctx.info.source.sender.to_string())
        } else {
            None
        },
    };

    let reaction = wa::Message {
        reaction_message: Some(wa::message::ReactionMessage {
            key: Some(key),
            text: Some(emoji.to_string()),
            sender_timestamp_ms: Some(chrono::Utc::now().timestamp_millis()),
            ..Default::default()
        }),
        ..Default::default()
    };

    if let Err(e) = ctx.send_message(reaction).await {
        tracing::error!("Failed to send reaction: {}", e);
    }
}

pub async fn send_reaction_to_message(ctx: &MessageContext, message_id: &str, emoji: &str) {
    let key = wa::MessageKey {
        remote_jid: Some(ctx.info.source.chat.to_string()),
        id: Some(message_id.to_string()),
        from_me: Some(false),
        participant: if ctx.info.source.is_group {
            Some(ctx.info.source.sender.to_string())
        } else {
            None
        },
    };

    let reaction = wa::Message {
        reaction_message: Some(wa::message::ReactionMessage {
            key: Some(key),
            text: Some(emoji.to_string()),
            sender_timestamp_ms: Some(chrono::Utc::now().timestamp_millis()),
            ..Default::default()
        }),
        ..Default::default()
    };

    if let Err(e) = ctx.send_message(reaction).await {
        tracing::error!("Failed to send reaction: {}", e);
    }
}

pub async fn handle_download(
    ctx: &MessageContext,
    media: &impl Downloadable,
    media_type: &str,
    extension: &str,
) {
    tracing::info!("Downloading incoming {}...", media_type);

    let mut buffer = Cursor::new(Vec::new());

    match ctx.client.download_to_file(media, &mut buffer).await {
        Ok(_) => {
            let data = buffer.into_inner();
            let filename = format!(
                "{}/{}_{}.{}",
                DOWNLOAD_DIR, media_type, ctx.info.id, extension
            );
            if let Err(e) = fs::write(&filename, data) {
                tracing::error!("Failed to save {} to disk: {}", media_type, e);
            } else {
                tracing::info!("Saved {} to {}", media_type, filename);
            }
        }
        Err(e) => {
            tracing::error!("Failed to download {}: {}", media_type, e);
        }
    }
}

pub fn convert_audio_for_whatsapp(input_path: &Path) -> Result<PathBuf, String> {
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

pub async fn send_local_media(ctx: &MessageContext, filename: &str, media_type: MediaType) {
    let path = match prepare_media_path(filename, &media_type).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Media preparation failed: {}", e);
            send_text_reply(ctx, &e).await;
            return;
        }
    };

    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("Failed to read asset: {}", e);
            return;
        }
    };

    let uploaded = match ctx.client.upload(data, media_type).await {
        Ok(u) => u,
        Err(e) => {
            tracing::error!("Failed to upload media: {}", e);
            send_text_reply(ctx, "Error: Failed to upload media.").await;
            return;
        }
    };

    let message = match media_type {
        MediaType::Image => wa::Message {
            image_message: Some(Box::new(wa::message::ImageMessage {
                url: Some(uploaded.url),
                direct_path: Some(uploaded.direct_path),
                media_key: Some(uploaded.media_key),
                file_enc_sha256: Some(uploaded.file_enc_sha256),
                file_sha256: Some(uploaded.file_sha256),
                file_length: Some(uploaded.file_length),
                mimetype: Some("image/jpeg".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        },
        MediaType::Video => wa::Message {
            video_message: Some(Box::new(wa::message::VideoMessage {
                url: Some(uploaded.url),
                direct_path: Some(uploaded.direct_path),
                media_key: Some(uploaded.media_key),
                file_enc_sha256: Some(uploaded.file_enc_sha256),
                file_sha256: Some(uploaded.file_sha256),
                file_length: Some(uploaded.file_length),
                mimetype: Some("video/mp4".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        },
        MediaType::Audio => wa::Message {
            audio_message: Some(Box::new(wa::message::AudioMessage {
                url: Some(uploaded.url),
                direct_path: Some(uploaded.direct_path),
                media_key: Some(uploaded.media_key),
                file_enc_sha256: Some(uploaded.file_enc_sha256),
                file_sha256: Some(uploaded.file_sha256),
                file_length: Some(uploaded.file_length),
                mimetype: Some("audio/ogg; codecs=opus".to_string()),
                ptt: Some(true),
                seconds: Some(5),
                ..Default::default()
            })),
            ..Default::default()
        },
        _ => wa::Message::default(),
    };

    if let Err(e) = ctx.send_message(message).await {
        tracing::error!("Failed to send media: {}", e);
    }
}

async fn prepare_media_path(filename: &str, media_type: &MediaType) -> Result<PathBuf, String> {
    let path = PathBuf::from(ASSETS_DIR).join(filename);

    if !path.exists() {
        return Err(format!("Error: Asset '{}' not found on server.", filename));
    }

    if matches!(media_type, MediaType::Audio) {
        convert_audio_for_whatsapp(&path)
    } else {
        Ok(path)
    }
}
