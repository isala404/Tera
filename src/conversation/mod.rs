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
    let event_id = format!("msg_{}", Uuid::new_v4().simple());
    let ev = ConversationEvent::message(
        &event_id,
        chrono::Utc::now().timestamp_millis(),
        "assistant",
        Some(text.to_string()),
        reply_to_id,
        turn_id,
        vec![],
    );
    let pref = ProviderRef::whatsapp(&event_id, provider_msg_id, chat_jid, true);
    let saved = history_db
        .insert_event_full(ev, Some(&pref), Some(("sent", None)))?
        .expect("assistant message must insert");
    Ok(saved.id)
}
