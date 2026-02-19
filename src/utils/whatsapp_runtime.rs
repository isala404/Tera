#![allow(dead_code)]

use crate::utils::whatsapp_helpers::{ASSETS_DIR, DOWNLOAD_DIR};
use crate::utils::whatsapp_router::EventRouter;
use crate::utils::whatsapp_store::PostgresStore;
use forge::prelude::*;
use sqlx::PgPool;
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use whatsapp_rust::bot::Bot;
use whatsapp_rust_tokio_transport::TokioWebSocketTransportFactory;
use whatsapp_rust_ureq_http_client::UreqHttpClient;

const RESTART_DELAY_SECS: u64 = 5;

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
    if let Err(err) = fs::create_dir_all(ASSETS_DIR) {
        tracing::error!("Failed to create assets directory: {}", err);
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
