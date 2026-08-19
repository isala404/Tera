pub mod buffer;
pub mod engine;
pub mod phoenix;
pub mod renderer;
pub mod session;
pub mod typing;

pub use engine::TurnEngine;
pub use phoenix::Phoenix;
pub use session::ConversationSession;

use crate::history::db::{ConversationEvent, HistoryDb, ProviderRef};
use anyhow::Result;
use uuid::Uuid;

/// Write something we just said into canonical history.
///
/// Shared by the turn engine and by Phoenix, which both have to leave the same
/// trail. The reply target is our own event id for the message being answered,
/// not the WhatsApp id: provider ids belong in `provider_refs` and nowhere else.
pub fn record_assistant_message(
    history_db: &HistoryDb,
    chat_jid: &str,
    provider_msg_id: &str,
    text: &str,
    turn_id: Option<String>,
    reply_to_id: Option<String>,
) -> Result<String> {
    let saved = history_db.insert_event(ConversationEvent {
        seq: None,
        id: format!("msg_{}", Uuid::new_v4().simple()),
        occurred_at_ms: chrono::Utc::now().timestamp_millis(),
        kind: "message".to_string(),
        actor: "assistant".to_string(),
        text: Some(text.to_string()),
        reply_to_id,
        turn_id,
        reaction_target_id: None,
        reaction_emoji: None,
        attachments: vec![],
    })?;
    history_db.record_provider_ref(&ProviderRef::whatsapp(
        &saved.id,
        provider_msg_id,
        chat_jid,
        true,
    ))?;
    history_db.record_delivery_event(&saved.id, "sent", None)?;
    Ok(saved.id)
}
