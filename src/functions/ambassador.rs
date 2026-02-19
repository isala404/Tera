use crate::schema::{MessageDirection, MessageStatus, WhatsappMessage};
use crate::utils::whatsapp_runtime::spawn_whatsapp_gateway;
use forge::prelude::*;
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use uuid::Uuid;

struct ChatBuffer {
    message_ids: Vec<Uuid>,
    last_activity: Instant,
    is_typing: bool,
}

#[forge::daemon(startup_delay = "5s")]
pub async fn ambassador(ctx: &DaemonContext) -> Result<()> {
    tracing::info!("Ambassador daemon starting - WhatsApp gateway initialized");

    let chat_buffers: Arc<RwLock<HashMap<String, ChatBuffer>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let mut whatsapp_gateway_task = spawn_whatsapp_gateway(Arc::new(ctx.db().clone()));

    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                // Check for typing timeouts and flush buffers
                if let Err(e) = process_chat_buffers(ctx, &chat_buffers).await {
                    tracing::error!("Failed to process chat buffers: {}", e);
                }

                // Also process any pending messages in the database
                if let Err(e) = process_pending_messages(ctx).await {
                    tracing::error!("Failed to process pending messages: {}", e);
                }
            }
            gateway_result = &mut whatsapp_gateway_task => {
                tracing::error!("WhatsApp gateway task stopped unexpectedly: {:?}", gateway_result);
                break;
            }
            _ = ctx.shutdown_signal() => {
                tracing::info!("Ambassador daemon shutting down gracefully");
                whatsapp_gateway_task.abort();
                // Flush all buffers before shutdown
                let buffers = chat_buffers.read().await;
                for (_chat_id, buffer) in buffers.iter() {
                    if let Err(e) = process_buffered_messages(ctx, &buffer.message_ids).await {
                        tracing::error!("Failed to process buffered messages during shutdown: {}", e);
                    }
                }
                break;
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
async fn process_chat_buffers(
    ctx: &DaemonContext,
    buffers: &Arc<RwLock<HashMap<String, ChatBuffer>>>,
) -> Result<()> {
    let mut chat_buffers = buffers.write().await;
    let now = Instant::now();
    const TYPING_TIMEOUT: Duration = Duration::from_secs(5);

    let chats_to_flush: Vec<String> = chat_buffers
        .iter()
        .filter_map(|(chat_id, buffer)| {
            // Flush if: typing stopped (elapsed > 5s) OR buffer has messages and timeout exceeded
            if !buffer.is_typing && now.duration_since(buffer.last_activity) > TYPING_TIMEOUT {
                Some(chat_id.clone())
            } else {
                None
            }
        })
        .collect();

    for chat_id in chats_to_flush {
        if let Some(buffer) = chat_buffers.remove(&chat_id)
            && let Err(e) = process_buffered_messages(ctx, &buffer.message_ids).await
        {
            tracing::error!(
                "Failed to process buffered messages from {}: {}",
                chat_id,
                e
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
async fn process_buffered_messages(ctx: &DaemonContext, message_ids: &[Uuid]) -> Result<()> {
    let db = ctx.db();

    for message_id in message_ids {
        let row = sqlx::query(
            r#"
            SELECT
                id, chat_id, direction, agent_id, status, content_text,
                media, embedding, metadata, created_at, updated_at
            FROM whatsapp_messages
            WHERE id = $1
            "#,
        )
        .bind(message_id)
        .fetch_optional(db)
        .await?;

        if let Some(row) = row {
            let message = WhatsappMessage {
                id: row.get("id"),
                chat_id: row.get("chat_id"),
                direction: parse_direction(&row.get::<String, _>("direction")),
                agent_id: row.get("agent_id"),
                status: parse_status(&row.get::<String, _>("status")),
                content_text: row.get("content_text"),
                media: row.get("media"),
                embedding: row.get("embedding"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            };

            if let Err(e) = handle_message(ctx, message).await {
                tracing::error!("Failed to handle buffered message {}: {}", message_id, e);
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
async fn process_pending_messages(ctx: &DaemonContext) -> Result<()> {
    let db = ctx.db();

    let rows = sqlx::query(
        r#"
        SELECT
            id, chat_id, direction, agent_id, status, content_text,
            media, embedding, metadata, created_at, updated_at
        FROM whatsapp_messages
        WHERE status = 'pending'
        LIMIT 100
        "#,
    )
    .fetch_all(db)
    .await?;

    for row in rows {
        let message = WhatsappMessage {
            id: row.get("id"),
            chat_id: row.get("chat_id"),
            direction: parse_direction(&row.get::<String, _>("direction")),
            agent_id: row.get("agent_id"),
            status: parse_status(&row.get::<String, _>("status")),
            content_text: row.get("content_text"),
            media: row.get("media"),
            embedding: row.get("embedding"),
            metadata: row.get("metadata"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        };

        if let Err(e) = handle_message(ctx, message).await {
            tracing::error!("Failed to handle message: {}", e);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn parse_direction(s: &str) -> MessageDirection {
    match s {
        "in" => MessageDirection::In,
        "out" => MessageDirection::Out,
        _ => MessageDirection::In,
    }
}

#[allow(dead_code)]
fn parse_status(s: &str) -> MessageStatus {
    match s {
        "pending_agent" => MessageStatus::PendingAgent,
        "sent_agent" => MessageStatus::SentAgent,
        "pending_gateway" => MessageStatus::PendingGateway,
        "sent_gateway" => MessageStatus::SentGateway,
        "failed_gateway" => MessageStatus::FailedGateway,
        _ => MessageStatus::PendingAgent,
    }
}

#[allow(dead_code)]
async fn handle_message(ctx: &DaemonContext, mut message: WhatsappMessage) -> Result<()> {
    // Route message based on direction
    match message.direction {
        MessageDirection::In => {
            // Inbound: mark as sent_agent (message ready for agent to process)
            message.status = MessageStatus::SentAgent;
            update_message_status(ctx, &message).await?;

            tracing::info!(
                "Processed inbound message from chat: {} ({})",
                message.chat_id,
                message.id
            );
        }
        MessageDirection::Out => {
            // Outbound: mark as sent_gateway (message sent to WhatsApp)
            message.status = MessageStatus::SentGateway;
            update_message_status(ctx, &message).await?;

            tracing::info!(
                "Processed outbound message to chat: {} ({})",
                message.chat_id,
                message.id
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
async fn update_message_status(ctx: &DaemonContext, message: &WhatsappMessage) -> Result<()> {
    let db = ctx.db();

    sqlx::query(
        r#"
        UPDATE whatsapp_messages
        SET status = $1, updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(format!("{:?}", message.status).to_lowercase())
    .bind(message.id)
    .execute(db)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_direction, parse_status};
    use crate::schema::{MessageDirection, MessageStatus};
    use crate::utils::test_db::setup::{init_test_db_with_vector, is_pgvector_unavailable};
    use serde_json::json;
    use sqlx::Row;
    use uuid::Uuid;

    async fn setup_db() -> Option<(forge::testing::IsolatedTestDb, sqlx::PgPool)> {
        match init_test_db_with_vector("ambassador_test").await {
            Ok(db) => {
                let pool = db.pool().clone();
                Some((db, pool))
            }
            Err(err) if is_pgvector_unavailable(err.as_ref()) => {
                eprintln!("Skipping ambassador DB test: {}", err);
                None
            }
            Err(err) => panic!("Failed to initialize test database: {}", err),
        }
    }

    #[tokio::test]
    async fn test_message_direction_parsing() {
        assert!(matches!(parse_direction("in"), MessageDirection::In));
        assert!(matches!(parse_direction("out"), MessageDirection::Out));
    }

    #[tokio::test]
    async fn test_message_status_parsing() {
        assert!(matches!(
            parse_status("pending_agent"),
            MessageStatus::PendingAgent
        ));
        assert!(matches!(
            parse_status("sent_agent"),
            MessageStatus::SentAgent
        ));
        assert!(matches!(
            parse_status("pending_gateway"),
            MessageStatus::PendingGateway
        ));
        assert!(matches!(
            parse_status("sent_gateway"),
            MessageStatus::SentGateway
        ));
        assert!(matches!(
            parse_status("failed_gateway"),
            MessageStatus::FailedGateway
        ));
    }

    #[tokio::test]
    async fn test_insert_message_into_database() {
        let Some((_db, pool)) = setup_db().await else {
            return;
        };
        let message_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO whatsapp_messages (
                id, chat_id, direction, status, content_text, metadata
            ) VALUES ($1, $2, $3::message_direction, $4::message_status, $5, $6)
            "#,
        )
        .bind(message_id)
        .bind("test_chat_123")
        .bind("in")
        .bind("pending_agent")
        .bind(Some("Hello, world!"))
        .bind(json!({"whatsapp_id": "wamsg123"}))
        .execute(&pool)
        .await
        .expect("Failed to insert test message");

        let result = sqlx::query("SELECT status::TEXT FROM whatsapp_messages WHERE id = $1")
            .bind(message_id)
            .fetch_one(&pool)
            .await
            .expect("Failed to fetch message");

        let status: String = result.get("status");
        assert_eq!(status, "pending_agent");
    }

    #[tokio::test]
    async fn test_message_with_agent_id() {
        let Some((_db, pool)) = setup_db().await else {
            return;
        };
        let agent_id = Uuid::new_v4();
        let message_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO whatsapp_messages (
                id, chat_id, direction, agent_id, status, content_text, metadata
            ) VALUES ($1, $2, $3::message_direction, $4, $5::message_status, $6, $7)
            "#,
        )
        .bind(message_id)
        .bind("test_chat_agent")
        .bind("in")
        .bind(agent_id)
        .bind("sent_agent")
        .bind(Some("Routed to agent"))
        .bind(json!({"agent": agent_id.to_string()}))
        .execute(&pool)
        .await
        .expect("Failed to insert message with agent");

        let result = sqlx::query("SELECT agent_id FROM whatsapp_messages WHERE id = $1")
            .bind(message_id)
            .fetch_one(&pool)
            .await
            .expect("Failed to fetch message");

        let fetched_agent_id: Option<Uuid> = result.get("agent_id");
        assert_eq!(fetched_agent_id, Some(agent_id));
    }
}
