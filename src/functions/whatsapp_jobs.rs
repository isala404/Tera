use crate::utils::gateway::{send_composing, send_paused};
use forge::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::Row;
use tokio::time::{Duration, sleep};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessWhatsappMessageJobInput {
    pub message_id: Uuid,
}

#[derive(Debug, Clone, PartialEq)]
enum AgentAction {
    Text(String),
    Reaction { emoji: String },
    AssetMedia { asset: String, media_type: String },
}

#[forge::job(
    public,
    timeout = "2m",
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
        text_preview = inbound.content_text.as_deref().unwrap_or(""),
        "Loaded inbound message"
    );

    let cancel_typing = CancellationToken::new();
    let typing_task = spawn_typing_heartbeat(inbound.chat_id.clone(), cancel_typing.clone());

    let delay_secs = {
        let mut rng = rand::rng();
        rng.random_range(5..=10)
    };
    sleep(Duration::from_secs(delay_secs)).await;

    match plan_agent_action(inbound.content_text.as_deref()) {
        Some(action) => {
            tracing::info!(%message_id, ?action, "Planned agent action");
            insert_outbound_action(ctx.db(), &inbound, action).await?;
            tracing::info!(%message_id, "Outbound action inserted");
        }
        None => {
            tracing::info!(%message_id, "No action planned for message");
        }
    }

    cancel_typing.cancel();
    let _ = typing_task.await;
    let _ = send_paused(&inbound.chat_id).await;

    Ok(())
}

struct InboundMessage {
    chat_id: String,
    content_text: Option<String>,
    metadata: Value,
}

async fn load_inbound_message(db: &sqlx::PgPool, message_id: Uuid) -> Result<InboundMessage> {
    let row = sqlx::query(
        r#"
        SELECT chat_id, content_text, metadata
        FROM whatsapp_messages
        WHERE id = $1 AND direction = 'in'::message_direction
        "#,
    )
    .bind(message_id)
    .fetch_optional(db)
    .await?
    .ok_or_else(|| ForgeError::NotFound(format!("Inbound message not found: {}", message_id)))?;

    Ok(InboundMessage {
        chat_id: row.get("chat_id"),
        content_text: row.get("content_text"),
        metadata: row.get("metadata"),
    })
}

fn plan_agent_action(content_text: Option<&str>) -> Option<AgentAction> {
    let text = content_text?.trim();
    let lower = text.to_lowercase();

    match lower.as_str() {
        "ping" => Some(AgentAction::Text("pong".to_string())),
        "echo" => Some(AgentAction::Text(
            "To echo text, use 'echo <text>'".to_string(),
        )),
        "react" => Some(AgentAction::Reaction {
            emoji: "👍".to_string(),
        }),
        "sendimage" => Some(AgentAction::AssetMedia {
            asset: "sample.jpg".to_string(),
            media_type: "image".to_string(),
        }),
        "sendvideo" => Some(AgentAction::AssetMedia {
            asset: "sample.mp4".to_string(),
            media_type: "video".to_string(),
        }),
        "sendvoice" => Some(AgentAction::AssetMedia {
            asset: "sample.ogg".to_string(),
            media_type: "audio".to_string(),
        }),
        _ if lower.starts_with("echo ") && text.len() > 5 => {
            Some(AgentAction::Text(text[5..].to_string()))
        }
        _ => None,
    }
}

async fn insert_outbound_action(
    db: &sqlx::PgPool,
    inbound: &InboundMessage,
    action: AgentAction,
) -> Result<()> {
    let outbound_id = Uuid::new_v4();

    let (content_text, media) = match action {
        AgentAction::Text(text) => (Some(text), None),
        AgentAction::Reaction { emoji } => {
            let participant = inbound
                .metadata
                .get("is_group")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then(|| inbound.metadata.get("sender_jid").and_then(Value::as_str))
                .flatten();
            let media = json!({
                "kind": "reaction",
                "emoji": emoji,
                "in_reply_to_message_id": inbound.metadata.get("whatsapp_message_id").and_then(Value::as_str),
                "participant": participant,
            });
            (None, Some(media))
        }
        AgentAction::AssetMedia { asset, media_type } => {
            let media = json!({
                "kind": "asset_media",
                "asset": asset,
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
            $1,
            $2,
            'out'::message_direction,
            'pending_gateway'::message_status,
            $3,
            $4,
            $5
        )
        "#,
    )
    .bind(outbound_id)
    .bind(&inbound.chat_id)
    .bind(content_text)
    .bind(media)
    .bind(json!({
        "generated_by": "process_whatsapp_message_job",
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
    use super::{AgentAction, plan_agent_action};

    #[test]
    fn test_plan_agent_action_commands() {
        assert_eq!(
            plan_agent_action(Some("ping")),
            Some(AgentAction::Text("pong".to_string()))
        );
        assert_eq!(
            plan_agent_action(Some("react")),
            Some(AgentAction::Reaction {
                emoji: "👍".to_string()
            })
        );

        assert_eq!(
            plan_agent_action(Some("sendimage")),
            Some(AgentAction::AssetMedia {
                asset: "sample.jpg".to_string(),
                media_type: "image".to_string(),
            })
        );
    }

    #[test]
    fn test_plan_agent_action_echo_text() {
        assert_eq!(
            plan_agent_action(Some("echo hello world")),
            Some(AgentAction::Text("hello world".to_string()))
        );
        assert!(plan_agent_action(Some("unknown command")).is_none());
    }

    #[test]
    fn test_plan_agent_action_none_for_empty() {
        assert!(plan_agent_action(None).is_none());
    }
}
