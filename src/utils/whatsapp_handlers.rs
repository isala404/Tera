#![allow(dead_code)]

use rand::Rng;
use sqlx::PgPool;
use std::time::{Duration, Instant};
use tokio::select;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use wacore::download::MediaType;
use wacore::proto_helpers::MessageExt;
use whatsapp_rust::bot::MessageContext;

use crate::utils::whatsapp_helpers::{
    handle_download, send_local_media, send_reaction, send_text_reply, simulate_recording,
    simulate_typing,
};
use crate::utils::whatsapp_task::ProcessingResult;

const ADJECTIVES: &[&str] = &[
    "HAPPY", "LUCKY", "SUNNY", "FAST", "BRIGHT", "COOL", "MAGIC", "SUPER", "BLUE", "RED", "GREEN",
    "GOLD", "SILVER", "BRAVE", "CALM", "WISE",
];
const NOUNS: &[&str] = &[
    "PANDA", "TIGER", "EAGLE", "LION", "MOON", "STAR", "OCEAN", "RIVER", "BEAR", "WOLF", "FOX",
    "HAWK", "SHIP", "PLANET", "FOREST", "MTN",
];

pub fn generate_pairing_code() -> String {
    let mut rng = rand::rng();
    let adj = ADJECTIVES[rng.random_range(0..ADJECTIVES.len())];
    let noun = NOUNS[rng.random_range(0..NOUNS.len())];
    format!("{} {}", adj, noun)
}

pub enum AuthResult {
    NewUser,
    PendingVerification { expired: bool },
    Authenticated,
}

pub async fn check_user_auth(db: &PgPool, sender_jid: &str) -> AuthResult {
    let row: Option<(bool, i64)> =
        match sqlx::query_as("SELECT is_authenticated, created_at FROM users WHERE jid = $1")
            .bind(sender_jid)
            .fetch_optional(db)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to query user: {}", e);
                return AuthResult::NewUser;
            }
        };

    match row {
        None => AuthResult::NewUser,
        Some((false, created_at)) => {
            let expired = (chrono::Utc::now().timestamp() - created_at) > 180;
            AuthResult::PendingVerification { expired }
        }
        Some((true, _)) => AuthResult::Authenticated,
    }
}

pub async fn create_new_user(db: &PgPool, sender_jid: &str) -> Option<String> {
    let code = generate_pairing_code();
    let now = chrono::Utc::now().timestamp();

    if let Err(e) = sqlx::query(
        "INSERT INTO users (jid, pairing_code, is_authenticated, created_at, last_seen) VALUES ($1, $2, FALSE, $3, $4)",
    )
    .bind(sender_jid)
    .bind(&code)
    .bind(now)
    .bind(now)
    .execute(db)
    .await
    {
        tracing::error!("Failed to create user: {}", e);
        return None;
    }

    tracing::info!(
        "New user verification required: jid={} code={}",
        sender_jid,
        code
    );
    Some(code)
}

pub async fn regenerate_pairing_code(db: &PgPool, sender_jid: &str) -> Option<String> {
    let new_code = generate_pairing_code();
    let now = chrono::Utc::now().timestamp();

    if let Err(e) =
        sqlx::query("UPDATE users SET pairing_code = $1, created_at = $2 WHERE jid = $3")
            .bind(&new_code)
            .bind(now)
            .bind(sender_jid)
            .execute(db)
            .await
    {
        tracing::error!("Failed to regenerate pairing code: {}", e);
        return None;
    }

    tracing::info!("Code expired: jid={} new_code={}", sender_jid, new_code);
    Some(new_code)
}

pub async fn get_stored_pairing_code(db: &PgPool, sender_jid: &str) -> Option<String> {
    sqlx::query_scalar("SELECT pairing_code FROM users WHERE jid = $1")
        .bind(sender_jid)
        .fetch_optional(db)
        .await
        .ok()
        .flatten()
}

pub async fn authenticate_user(db: &PgPool, sender_jid: &str) {
    if let Err(e) =
        sqlx::query("UPDATE users SET is_authenticated = TRUE, last_seen = $1 WHERE jid = $2")
            .bind(chrono::Utc::now().timestamp())
            .bind(sender_jid)
            .execute(db)
            .await
    {
        tracing::error!("Failed to authenticate user: {}", e);
    }
}

pub async fn update_last_seen(db: &PgPool, sender_jid: &str) {
    let _ = sqlx::query("UPDATE users SET last_seen = $1 WHERE jid = $2")
        .bind(chrono::Utc::now().timestamp())
        .bind(sender_jid)
        .execute(db)
        .await;
}

pub async fn handle_new_user(ctx: &MessageContext) {
    simulate_typing(ctx).await;
    send_text_reply(
        ctx,
        "Hello! I'm the Hospital Support Bot.\n\nTo continue, I need to verify it's really you. Please check the server logs and reply with the two words (Pairing Code) you see there.\n\n(The code is valid for 3 minutes)",
    )
    .await;
}

pub async fn handle_expired_code(ctx: &MessageContext) {
    simulate_typing(ctx).await;
    send_text_reply(
        ctx,
        "Your previous pairing code has expired.\n\nI've generated a new one. Please check the server logs again and send the new two words.",
    )
    .await;
}

pub async fn handle_verification_attempt(
    ctx: &MessageContext,
    db: &PgPool,
    sender_jid: &str,
    input_text: &str,
) -> bool {
    let stored_code = match get_stored_pairing_code(db, sender_jid).await {
        Some(c) => c,
        None => return false,
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
            "Awesome! You are now verified.\n\nYou can use the following commands:\n- *ping*: Check if I'm alive\n- *sendimage*: Get a sample image\n- *sendvideo*: Get a sample video\n- *sendvoice*: Get a sample voice note\n- *react*: I'll give you a thumbs up",
        )
        .await;
        true
    } else {
        simulate_typing(ctx).await;
        send_text_reply(
            ctx,
            "That doesn't look like the correct code. Please check the logs and try again.",
        )
        .await;
        false
    }
}

pub async fn process_with_cancellation(
    ctx: MessageContext,
    cancel_token: CancellationToken,
) -> Option<ProcessingResult> {
    let started_at = Instant::now();
    let delay_secs = {
        let mut rng = rand::rng();
        rng.random_range(5..=10)
    };
    let delay = Duration::from_secs(delay_secs);

    tracing::info!(
        "Processing message {} with {}s delay",
        ctx.info.id,
        delay_secs
    );

    let _ = ctx
        .client
        .chatstate()
        .send_composing(&ctx.info.source.chat)
        .await;

    select! {
        _ = cancel_token.cancelled() => {
            tracing::info!("Processing cancelled: message_id={}", ctx.info.id);
            return None;
        }
        _ = sleep(delay) => {}
    }

    if cancel_token.is_cancelled() {
        return None;
    }

    let response = process_message(&ctx).await;

    tracing::info!(
        "Completed processing: message_id={} elapsed={:?}",
        ctx.info.id,
        started_at.elapsed()
    );

    Some(ProcessingResult {
        response_text: response.unwrap_or_else(|| "Processed".to_string()),
    })
}

async fn process_message(ctx: &MessageContext) -> Option<String> {
    download_media_if_present(ctx).await;
    handle_text_command(ctx).await
}

async fn download_media_if_present(ctx: &MessageContext) {
    let base_msg = ctx.message.get_base_message();

    if let Some(image) = &base_msg.image_message {
        handle_download(ctx, &**image, "image", "jpg").await;
    } else if let Some(video) = &base_msg.video_message {
        handle_download(ctx, &**video, "video", "mp4").await;
    } else if let Some(audio) = &base_msg.audio_message {
        let ext = if audio.ptt.unwrap_or(false) {
            "ogg"
        } else {
            "mp3"
        };
        handle_download(ctx, &**audio, "audio", ext).await;
    } else if let Some(sticker) = &base_msg.sticker_message {
        handle_download(ctx, &**sticker, "sticker", "webp").await;
    }
}

async fn handle_text_command(ctx: &MessageContext) -> Option<String> {
    let text = ctx.message.text_content()?;
    let command_raw = text.trim();
    let command_lower = command_raw.to_lowercase();

    match command_lower.as_str() {
        "ping" => {
            simulate_typing(ctx).await;
            send_text_reply(ctx, "pong").await;
            Some("pong".to_string())
        }
        "echo" => {
            handle_echo_command(ctx).await;
            Some("echo".to_string())
        }
        "react" => {
            send_reaction(ctx, "\u{1f44d}").await;
            Some("react".to_string())
        }
        "sendimage" => {
            simulate_typing(ctx).await;
            send_local_media(ctx, "sample.jpg", MediaType::Image).await;
            Some("sendimage".to_string())
        }
        "sendvideo" => {
            simulate_typing(ctx).await;
            send_local_media(ctx, "sample.mp4", MediaType::Video).await;
            Some("sendvideo".to_string())
        }
        "sendvoice" => {
            simulate_recording(ctx).await;
            send_local_media(ctx, "sample.ogg", MediaType::Audio).await;
            Some("sendvoice".to_string())
        }
        _ if command_lower.starts_with("echo ") && command_raw.len() > 5 => {
            let content = &command_raw[5..];
            simulate_typing(ctx).await;
            send_text_reply(ctx, content).await;
            Some(format!("echo: {}", content))
        }
        _ => None,
    }
}

async fn handle_echo_command(ctx: &MessageContext) {
    if let Some(extended) = &ctx.message.extended_text_message
        && let Some(context) = &extended.context_info
        && let Some(quoted) = &context.quoted_message
    {
        let msg_to_send = *quoted.clone();
        if let Err(e) = ctx.send_message(msg_to_send).await {
            tracing::error!("Failed to echo message: {}", e);
        }
        return;
    }

    simulate_typing(ctx).await;
    send_text_reply(
        ctx,
        "To echo a message, reply to it with 'echo'. To echo text, use 'echo <text>'.",
    )
    .await;
}

pub async fn replay_cached_response(ctx: &MessageContext, result: &ProcessingResult) {
    simulate_typing(ctx).await;
    send_text_reply(ctx, &format!("[Cached] {}", result.response_text)).await;
}

#[cfg(test)]
mod tests {
    use super::generate_pairing_code;

    #[test]
    fn test_generate_pairing_code_format() {
        let code = generate_pairing_code();
        let parts: Vec<&str> = code.split_whitespace().collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].chars().all(|c| c.is_ascii_uppercase()));
        assert!(parts[1].chars().all(|c| c.is_ascii_uppercase()));
    }
}
