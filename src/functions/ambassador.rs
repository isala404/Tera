use crate::utils::gateway::{
    media_json_kind, send_asset_media_message, send_reaction_message, send_text_message,
    spawn_whatsapp_gateway,
};
use forge::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::Row;
use std::sync::Arc;
use tokio::time::Duration;
use uuid::Uuid;

const LOOP_INTERVAL_MS: u64 = 500;
const BATCH_LIMIT: i64 = 25;

#[derive(Debug, Deserialize)]
struct RpcError {
    code: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct RpcResponse {
    success: bool,
    data: Option<Value>,
    error: Option<RpcError>,
}

#[forge::daemon(startup_delay = "5s")]
pub async fn ambassador(ctx: &DaemonContext) -> Result<()> {
    tracing::info!("Ambassador daemon starting");

    let http = reqwest::Client::new();
    let mut whatsapp_gateway_task = spawn_whatsapp_gateway(Arc::new(ctx.db().clone()));

    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(LOOP_INTERVAL_MS)) => {
                if let Err(err) = dispatch_pending_inbound_jobs(ctx, &http).await {
                    tracing::error!("Inbound dispatch loop failed: {}", err);
                }

                if let Err(err) = deliver_pending_outbound_messages(ctx).await {
                    tracing::error!("Outbound delivery loop failed: {}", err);
                }
            }
            gateway_result = &mut whatsapp_gateway_task => {
                tracing::error!("WhatsApp gateway task stopped: {:?}", gateway_result);
                break;
            }
            _ = ctx.shutdown_signal() => {
                tracing::info!("Ambassador daemon shutting down");
                whatsapp_gateway_task.abort();
                break;
            }
        }
    }

    Ok(())
}

async fn dispatch_pending_inbound_jobs(ctx: &DaemonContext, http: &reqwest::Client) -> Result<()> {
    let rows = sqlx::query(
        r#"
        SELECT id
        FROM whatsapp_messages
        WHERE direction = 'in'::message_direction
          AND status = 'pending_agent'::message_status
        ORDER BY created_at ASC
        LIMIT $1
        "#,
    )
    .bind(BATCH_LIMIT)
    .fetch_all(ctx.db())
    .await?;

    if !rows.is_empty() {
        tracing::info!(count = rows.len(), "Processing pending inbound messages");
    }

    for row in rows {
        let message_id: Uuid = row.get("id");

        match dispatch_processing_job(http, message_id).await {
            Ok(job_id) => {
                tracing::info!(%message_id, %job_id, "Dispatched processing job");
                update_status(
                    ctx.db(),
                    message_id,
                    "sent_agent",
                    json!({
                        "job_id": job_id,
                        "job_dispatched_at": chrono::Utc::now(),
                    }),
                )
                .await?;
            }
            Err(err) => {
                tracing::error!(%message_id, error = %err, "Failed to dispatch processing job");
            }
        }
    }

    Ok(())
}

async fn dispatch_processing_job(http: &reqwest::Client, message_id: Uuid) -> Result<String> {
    let endpoint = internal_rpc_endpoint("process_whatsapp_message_job");

    let response = http
        .post(endpoint)
        .json(&json!({
            "args": {
                "message_id": message_id,
            }
        }))
        .send()
        .await
        .map_err(|e| ForgeError::Internal(format!("Failed to call internal RPC: {}", e)))?;

    let status = response.status();
    let payload: RpcResponse = response
        .json()
        .await
        .map_err(|e| ForgeError::Internal(format!("Invalid internal RPC response: {}", e)))?;

    if !status.is_success() || !payload.success {
        let err = payload
            .error
            .map(|e| format!("{}: {}", e.code, e.message))
            .unwrap_or_else(|| "unknown error".to_string());
        return Err(ForgeError::Internal(format!(
            "Job dispatch RPC failed for {}: {}",
            message_id, err
        )));
    }

    let job_id = payload
        .data
        .as_ref()
        .and_then(|v| v.get("job_id"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    Ok(job_id)
}

async fn deliver_pending_outbound_messages(ctx: &DaemonContext) -> Result<()> {
    let rows = sqlx::query(
        r#"
        SELECT id, chat_id, content_text, media, metadata
        FROM whatsapp_messages
        WHERE direction = 'out'::message_direction
          AND status = 'pending_gateway'::message_status
        ORDER BY created_at ASC
        LIMIT $1
        "#,
    )
    .bind(BATCH_LIMIT)
    .fetch_all(ctx.db())
    .await?;

    if !rows.is_empty() {
        tracing::info!(count = rows.len(), "Delivering pending outbound messages");
    }

    for row in rows {
        let id: Uuid = row.get("id");
        let chat_id: String = row.get("chat_id");
        let content_text: Option<String> = row.get("content_text");
        let media: Option<Value> = row.get("media");

        let kind = media
            .as_ref()
            .and_then(|m| m.get("kind"))
            .and_then(Value::as_str)
            .unwrap_or("text");
        tracing::info!(%id, %chat_id, %kind, "Delivering outbound message");

        match deliver_outbound_message(&chat_id, content_text.as_deref(), media.as_ref()).await {
            Ok(gateway_message_id) => {
                tracing::info!(%id, %gateway_message_id, "Outbound message delivered");
                update_status(
                    ctx.db(),
                    id,
                    "sent_gateway",
                    json!({
                        "gateway_message_id": gateway_message_id,
                        "delivered_at": chrono::Utc::now(),
                    }),
                )
                .await?;
            }
            Err(err) => {
                tracing::error!(%id, %chat_id, error = %err, "Outbound delivery failed");
                update_status(
                    ctx.db(),
                    id,
                    "failed_gateway",
                    json!({
                        "delivery_error": err.to_string(),
                        "failed_at": chrono::Utc::now(),
                    }),
                )
                .await?;
            }
        }
    }

    Ok(())
}

async fn deliver_outbound_message(
    chat_id: &str,
    content_text: Option<&str>,
    media: Option<&Value>,
) -> Result<String> {
    if let Some(media_payload) = media {
        return deliver_media_message(chat_id, media_payload).await;
    }

    if let Some(text) = content_text {
        return send_text_message(chat_id, text).await;
    }

    Err(ForgeError::Validation(
        "Outbound message has neither content_text nor media".to_string(),
    ))
}

async fn deliver_media_message(chat_id: &str, media: &Value) -> Result<String> {
    match media_json_kind(media) {
        Some("reaction") => {
            let emoji = media.get("emoji").and_then(Value::as_str).unwrap_or("👍");
            let target_message_id = media
                .get("in_reply_to_message_id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ForgeError::Validation(
                        "Reaction media payload missing in_reply_to_message_id".to_string(),
                    )
                })?;
            let participant = media.get("participant").and_then(Value::as_str);

            send_reaction_message(chat_id, target_message_id, participant, emoji).await
        }
        Some("asset_media") => {
            let asset = media
                .get("asset")
                .and_then(Value::as_str)
                .ok_or_else(|| ForgeError::Validation("asset_media missing asset".to_string()))?;
            let media_type = media
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ForgeError::Validation("asset_media missing media_type".to_string())
                })?;

            send_asset_media_message(chat_id, asset, media_type).await
        }
        Some(other) => Err(ForgeError::Validation(format!(
            "Unsupported outbound media kind: {}",
            other
        ))),
        None => Err(ForgeError::Validation(
            "Outbound media payload missing kind".to_string(),
        )),
    }
}

async fn update_status(
    db: &sqlx::PgPool,
    id: Uuid,
    status: &str,
    metadata_patch: Value,
) -> Result<()> {
    sqlx::query(
        r#"
        UPDATE whatsapp_messages
        SET status = $1::message_status,
            metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb,
            updated_at = NOW()
        WHERE id = $3
        "#,
    )
    .bind(status)
    .bind(metadata_patch)
    .bind(id)
    .execute(db)
    .await?;

    Ok(())
}

fn internal_rpc_endpoint(function_name: &str) -> String {
    let base = std::env::var("FORGE_INTERNAL_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
        .trim_end_matches('/')
        .to_string();
    format!("{}/_api/rpc/{}", base, function_name)
}

#[cfg(test)]
mod tests {
    use super::internal_rpc_endpoint;

    #[test]
    fn test_internal_rpc_endpoint_default() {
        let endpoint = internal_rpc_endpoint("process_whatsapp_message_job");
        assert!(endpoint.ends_with("/_api/rpc/process_whatsapp_message_job"));
    }
}
