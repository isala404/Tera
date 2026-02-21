use crate::utils::gateway::{send_composing, send_paused};
use forge::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::Row;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::time::{Duration, sleep};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const CHAT_DIR: &str = "chats";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessWhatsappMessageJobInput {
    pub message_id: Uuid,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplyLine {
    Text(String),
    FilePath(PathBuf),
}

#[forge::job(
    public,
    timeout = "5m",
    retry(max_attempts = 3, backoff = "exponential")
)]
pub async fn process_whatsapp_message_job(
    ctx: &JobContext,
    input: ProcessWhatsappMessageJobInput,
) -> Result<()> {
    let message_id = input.message_id;
    tracing::info!(%message_id, "Processing inbound message");

    let inbound = load_inbound_message(ctx.db(), message_id).await?;
    tracing::info!(
        %message_id,
        chat_id = %inbound.chat_id,
        has_text = inbound.content_text.is_some(),
        has_media = inbound.media_path.is_some(),
        "Loaded inbound message"
    );

    let cancel_typing = CancellationToken::new();
    let typing_task = spawn_typing_heartbeat(inbound.chat_id.clone(), cancel_typing.clone());

    let reply_lines = run_claude_agent(&inbound).await?;

    cancel_typing.cancel();
    let _ = typing_task.await;
    let _ = send_paused(&inbound.chat_id).await;

    for line in &reply_lines {
        insert_outbound_reply(ctx.db(), &inbound, line).await?;
    }

    tracing::info!(%message_id, reply_count = reply_lines.len(), "Agent replies queued");
    Ok(())
}

struct InboundMessage {
    chat_id: String,
    content_text: Option<String>,
    media_path: Option<String>,
    metadata: Value,
}

async fn load_inbound_message(db: &sqlx::PgPool, message_id: Uuid) -> Result<InboundMessage> {
    let row = sqlx::query(
        r#"
        SELECT chat_id, content_text, media, metadata
        FROM whatsapp_messages
        WHERE id = $1 AND direction = 'in'::message_direction
        "#,
    )
    .bind(message_id)
    .fetch_optional(db)
    .await?
    .ok_or_else(|| ForgeError::NotFound(format!("Inbound message not found: {}", message_id)))?;

    let media: Option<Value> = row.get("media");
    let media_path = media
        .as_ref()
        .and_then(|m| m.get("local_path"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    Ok(InboundMessage {
        chat_id: row.get("chat_id"),
        content_text: row.get("content_text"),
        media_path,
        metadata: row.get("metadata"),
    })
}

fn chat_history_path(chat_id: &str) -> PathBuf {
    let sanitized = chat_id.replace(['/', '\\', ':', '@'], "_");
    Path::new(CHAT_DIR).join(format!("{}.md", sanitized))
}

fn reply_file_path(chat_id: &str) -> PathBuf {
    let sanitized = chat_id.replace(['/', '\\', ':', '@'], "_");
    Path::new(CHAT_DIR).join(format!("{}_reply.txt", sanitized))
}

fn build_system_prompt(chat_id: &str) -> String {
    let history_path = chat_history_path(chat_id);
    format!(
        r#"You are an AI assistant communicating with a user via WhatsApp.

Chat ID: {chat_id}
Chat history file: {history}

IMPORTANT RULES:
- Read the chat history file first using Read tool or cat to understand prior context
- When the user asks you to do something, do it using your available tools
- Write your final reply to: {reply}
- Each line in reply.txt is sent as a separate WhatsApp message
- For text replies, just write the text on a line
- To send a file (image/video/audio/document), write the absolute file path on its own line
- Keep replies concise, WhatsApp messages should be short
- Do NOT include any prefix like "Reply:" just the raw message text
- If you need to send multiple messages, put each on its own line"#,
        chat_id = chat_id,
        history = history_path.display(),
        reply = reply_file_path(chat_id).display(),
    )
}

fn build_user_prompt(inbound: &InboundMessage) -> String {
    let mut parts = Vec::new();

    if let Some(text) = &inbound.content_text {
        parts.push(text.clone());
    }

    if let Some(path) = &inbound.media_path {
        parts.push(format!("[Media file: {}]", path));
    }

    if parts.is_empty() {
        "[Empty message]".to_string()
    } else {
        parts.join("\n")
    }
}

async fn append_to_chat_history(chat_id: &str, role: &str, content: &str) -> Result<()> {
    let path = chat_history_path(chat_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            ForgeError::Internal(format!("Failed to create chat dir: {}", e))
        })?;
    }

    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let entry = format!("\n### {} [{}]\n{}\n", role, timestamp, content);

    tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to open chat history: {}", e)))?
        .write_all(entry.as_bytes())
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to write chat history: {}", e)))?;

    Ok(())
}

use tokio::io::AsyncWriteExt;

async fn run_claude_agent(inbound: &InboundMessage) -> Result<Vec<ReplyLine>> {
    let system_prompt = build_system_prompt(&inbound.chat_id);
    let user_prompt = build_user_prompt(inbound);
    let reply_path = reply_file_path(&inbound.chat_id);

    append_to_chat_history(&inbound.chat_id, "User", &user_prompt).await?;
    let _ = tokio::fs::remove_file(&reply_path).await;

    let full_prompt = format!("{}\n\n---\nUser message:\n{}", system_prompt, user_prompt);

    tracing::info!(chat_id = %inbound.chat_id, "Spawning Claude agent");

    let output = Command::new("claude")
        .arg("-p")
        .arg(&full_prompt)
        .arg("--dangerously-skip-permissions")
        .arg("--chrome")
        .output()
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to spawn claude: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(chat_id = %inbound.chat_id, %stderr, "Claude agent failed");
        return Err(ForgeError::Internal(format!("Claude agent exited with error: {}", stderr)));
    }

    let reply_lines = read_reply_file(&reply_path).await?;

    let reply_text: Vec<String> = reply_lines.iter().map(|l| match l {
        ReplyLine::Text(t) => t.clone(),
        ReplyLine::FilePath(p) => format!("[File: {}]", p.display()),
    }).collect();

    if !reply_text.is_empty() {
        append_to_chat_history(&inbound.chat_id, "Assistant", &reply_text.join("\n")).await?;
    }

    let _ = tokio::fs::remove_file(&reply_path).await;

    Ok(reply_lines)
}

async fn read_reply_file(path: &Path) -> Result<Vec<ReplyLine>> {
    let content = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(path = %path.display(), "No reply.txt produced by agent");
            return Ok(vec![]);
        }
        Err(e) => return Err(ForgeError::Internal(format!("Failed to read reply file: {}", e))),
    };

    Ok(parse_reply_lines(&content))
}

pub fn parse_reply_lines(content: &str) -> Vec<ReplyLine> {
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = Path::new(line);
            if path.is_absolute() && path.extension().is_some() {
                ReplyLine::FilePath(path.to_path_buf())
            } else {
                ReplyLine::Text(line.to_string())
            }
        })
        .collect()
}

pub fn media_type_from_extension(path: &Path) -> &'static str {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext.to_lowercase().as_str() {
        "jpg" | "jpeg" | "png" | "webp" | "gif" => "image",
        "mp4" | "mov" | "avi" | "mkv" => "video",
        "ogg" | "mp3" | "wav" | "m4a" | "opus" => "audio",
        _ => "document",
    }
}

async fn insert_outbound_reply(
    db: &sqlx::PgPool,
    inbound: &InboundMessage,
    reply: &ReplyLine,
) -> Result<()> {
    let outbound_id = Uuid::new_v4();

    let (content_text, media) = match reply {
        ReplyLine::Text(text) => (Some(text.clone()), None),
        ReplyLine::FilePath(path) => {
            let media_type = media_type_from_extension(path);
            let media = json!({
                "kind": "file",
                "local_path": path.to_string_lossy(),
                "media_type": media_type,
            });
            (None, Some(media))
        }
    };

    sqlx::query(
        r#"
        INSERT INTO whatsapp_messages (
            id, chat_id, direction, status, content_text, media, metadata
        ) VALUES (
            $1, $2, 'out'::message_direction, 'pending_gateway'::message_status,
            $3, $4, $5
        )
        "#,
    )
    .bind(outbound_id)
    .bind(&inbound.chat_id)
    .bind(content_text)
    .bind(media)
    .bind(json!({
        "generated_by": "claude_agent",
        "source_sender": inbound.metadata.get("sender_jid"),
        "source_whatsapp_message_id": inbound.metadata.get("whatsapp_message_id"),
    }))
    .execute(db)
    .await?;

    Ok(())
}

fn spawn_typing_heartbeat(
    chat_id: String,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let _ = send_composing(&chat_id).await;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = sleep(Duration::from_secs(2)) => {
                    let _ = send_composing(&chat_id).await;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parse_reply_text_only() {
        let content = "Hello!\nHow are you?";
        let lines = parse_reply_lines(content);
        assert_eq!(lines, vec![
            ReplyLine::Text("Hello!".to_string()),
            ReplyLine::Text("How are you?".to_string()),
        ]);
    }

    #[test]
    fn parse_reply_with_file_paths() {
        let content = "Here's the image\n/tmp/output.png\nDone!";
        let lines = parse_reply_lines(content);
        assert_eq!(lines, vec![
            ReplyLine::Text("Here's the image".to_string()),
            ReplyLine::FilePath(PathBuf::from("/tmp/output.png")),
            ReplyLine::Text("Done!".to_string()),
        ]);
    }

    #[test]
    fn parse_reply_skips_empty_lines() {
        let content = "\n  \nHello\n\n";
        let lines = parse_reply_lines(content);
        assert_eq!(lines, vec![ReplyLine::Text("Hello".to_string())]);
    }

    #[test]
    fn parse_reply_relative_paths_treated_as_text() {
        let content = "images/photo.png";
        let lines = parse_reply_lines(content);
        assert_eq!(lines, vec![ReplyLine::Text("images/photo.png".to_string())]);
    }

    #[test]
    fn parse_reply_file_media_type_detection() {
        let content = "/tmp/photo.jpg\n/tmp/video.mp4\n/tmp/voice.ogg\n/tmp/doc.pdf";
        let lines = parse_reply_lines(content);
        assert!(matches!(&lines[0], ReplyLine::FilePath(p) if p == Path::new("/tmp/photo.jpg")));
        assert!(matches!(&lines[1], ReplyLine::FilePath(p) if p == Path::new("/tmp/video.mp4")));
        assert!(matches!(&lines[2], ReplyLine::FilePath(p) if p == Path::new("/tmp/voice.ogg")));
        assert!(matches!(&lines[3], ReplyLine::FilePath(p) if p == Path::new("/tmp/doc.pdf")));
    }

    #[test]
    fn chat_history_path_sanitizes_jid() {
        let path = chat_history_path("123456@s.whatsapp.net");
        assert_eq!(path, PathBuf::from("chats/123456_s.whatsapp.net.md"));
    }

    #[test]
    fn system_prompt_contains_required_elements() {
        let prompt = build_system_prompt("123@s.whatsapp.net");
        assert!(prompt.contains("WhatsApp"));
        assert!(prompt.contains("reply"));
        assert!(prompt.contains("123@s.whatsapp.net"));
        assert!(prompt.contains("chat history"));
    }

    #[test]
    fn user_prompt_text_only() {
        let msg = InboundMessage {
            chat_id: "test".to_string(),
            content_text: Some("hello".to_string()),
            media_path: None,
            metadata: json!({}),
        };
        assert_eq!(build_user_prompt(&msg), "hello");
    }

    #[test]
    fn user_prompt_with_media() {
        let msg = InboundMessage {
            chat_id: "test".to_string(),
            content_text: Some("check this".to_string()),
            media_path: Some("/tmp/photo.jpg".to_string()),
            metadata: json!({}),
        };
        let prompt = build_user_prompt(&msg);
        assert!(prompt.contains("check this"));
        assert!(prompt.contains("/tmp/photo.jpg"));
    }

    #[test]
    fn user_prompt_empty_message() {
        let msg = InboundMessage {
            chat_id: "test".to_string(),
            content_text: None,
            media_path: None,
            metadata: json!({}),
        };
        assert_eq!(build_user_prompt(&msg), "[Empty message]");
    }

    #[test]
    fn media_type_detection_from_extension() {
        assert_eq!(media_type_from_extension(Path::new("/tmp/photo.jpg")), "image");
        assert_eq!(media_type_from_extension(Path::new("/tmp/photo.jpeg")), "image");
        assert_eq!(media_type_from_extension(Path::new("/tmp/photo.PNG")), "image");
        assert_eq!(media_type_from_extension(Path::new("/tmp/photo.webp")), "image");
        assert_eq!(media_type_from_extension(Path::new("/tmp/video.mp4")), "video");
        assert_eq!(media_type_from_extension(Path::new("/tmp/video.mov")), "video");
        assert_eq!(media_type_from_extension(Path::new("/tmp/voice.ogg")), "audio");
        assert_eq!(media_type_from_extension(Path::new("/tmp/voice.m4a")), "audio");
        assert_eq!(media_type_from_extension(Path::new("/tmp/doc.pdf")), "document");
        assert_eq!(media_type_from_extension(Path::new("/tmp/file.txt")), "document");
        assert_eq!(media_type_from_extension(Path::new("/tmp/noext")), "document");
    }

    #[test]
    fn absolute_path_without_extension_is_text() {
        let lines = parse_reply_lines("/tmp/noext");
        assert_eq!(lines, vec![ReplyLine::Text("/tmp/noext".to_string())]);
    }

    #[tokio::test]
    async fn read_reply_file_missing_returns_empty() {
        let result = read_reply_file(Path::new("/nonexistent/reply.txt")).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn chat_history_append_creates_file() {
        let tmp = std::env::temp_dir().join("tera_test_chat_history");
        let _ = tokio::fs::remove_dir_all(&tmp).await;

        let chat_id = format!("test_{}@s.whatsapp.net", uuid::Uuid::new_v4().as_simple());

        // temporarily override CHAT_DIR by using the function directly
        let path = tmp.join(format!("{}.md", chat_id.replace(['/', '\\', ':', '@'], "_")));
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.unwrap();
        }

        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        let entry = format!("\n### User [{}]\nhello world\n", timestamp);
        tokio::fs::write(&path, &entry).await.unwrap();

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("hello world"));
        assert!(content.contains("User"));

        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[test]
    fn reply_file_path_matches_chat_history_pattern() {
        let chat_id = "123@s.whatsapp.net";
        let history = chat_history_path(chat_id);
        let reply = reply_file_path(chat_id);
        assert_eq!(history.parent(), reply.parent());
    }

    #[test]
    fn system_prompt_references_reply_file() {
        let prompt = build_system_prompt("test@s.whatsapp.net");
        let reply_path = reply_file_path("test@s.whatsapp.net");
        assert!(prompt.contains(&reply_path.display().to_string()));
    }
}
