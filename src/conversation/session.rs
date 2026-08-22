//! Live conversation state shared between the turn engine and the MCP server.
//!
//! These two run in the same process but on different paths, the engine drives
//! turns from inbound WhatsApp messages, the MCP server serves tool calls that
//! Codex makes during those turns, and they need to agree on two things:
//!
//! * which chat the assistant is talking to, so a tool call replies into the
//!   conversation that prompted it rather than a statically configured number;
//! * whether the turn already spoke, so the final agent text is not delivered on
//!   top of a `send_message` the agent already made.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct ConversationSession {
    sends: Arc<AtomicU64>,
    chat_jid: Arc<Mutex<Option<String>>>,
    turn_id: Arc<Mutex<Option<String>>>,
}

impl ConversationSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember which chat the current conversation belongs to.
    ///
    /// Set from the inbound message rather than configuration: the owner's JID
    /// carries a device suffix that varies per device, and the configured owner
    /// value is a matching pattern, not a routable address.
    pub fn set_chat(&self, jid: &str) {
        *self.chat_jid.lock().unwrap() = Some(jid.to_string());
    }

    pub fn chat(&self) -> Option<String> {
        self.chat_jid.lock().unwrap().clone()
    }

    pub fn set_turn(&self, turn_id: Option<&str>) {
        *self.turn_id.lock().unwrap() = turn_id.map(str::to_string);
    }

    pub fn turn(&self) -> Option<String> {
        self.turn_id.lock().unwrap().clone()
    }

    /// Called after a `send_message` tool call actually reaches the provider.
    pub fn record_send(&self) {
        self.sends.fetch_add(1, Ordering::SeqCst);
    }

    /// Monotonic count; snapshot it before a turn and compare after.
    pub fn count(&self) -> u64 {
        self.sends.load(Ordering::SeqCst)
    }

    pub fn sends_since(&self, snapshot: u64) -> u64 {
        self.count().saturating_sub(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracks_sends_across_clones() {
        let session = ConversationSession::new();
        let snapshot = session.count();

        // The MCP server holds its own clone of the same session.
        let mcp_side = session.clone();
        mcp_side.record_send();
        mcp_side.record_send();

        assert_eq!(session.sends_since(snapshot), 2);
    }

    #[test]
    fn test_quiet_turn_reports_no_sends() {
        let session = ConversationSession::new();
        let snapshot = session.count();
        assert_eq!(session.sends_since(snapshot), 0);
    }

    #[test]
    fn test_chat_target_is_shared() {
        let session = ConversationSession::new();
        assert_eq!(session.chat(), None);

        session.set_chat("254910671147212:26@lid");
        assert_eq!(
            session.clone().chat().as_deref(),
            Some("254910671147212:26@lid")
        );
    }
}
