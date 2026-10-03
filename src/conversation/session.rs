//! Live conversation state shared between the turn engine and the MCP server.
//!
//! These two run in the same process but on different paths, the engine drives
//! turns from inbound WhatsApp messages, the MCP server serves tool calls that
//! Codex makes during those turns, and they need to agree on two things:
//!
//! * which chat the assistant is talking to, so a tool call replies into the
//!   conversation that prompted it rather than a statically configured number;
//! * whether a turn already spoke, so the final agent text is not delivered on
//!   top of a `send_message` the agent already made.
//!
//! Both are kept per worker. The foreground conversation, each scheduled task
//! and Phoenix recovery run at the same time, and one must not stamp or count
//! another's messages.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The worker id of the main conversation. Scheduled tasks and Phoenix name
/// themselves when their thread starts.
pub const FOREGROUND: &str = "foreground";

#[derive(Clone, Default)]
struct WorkerSession {
    turn_id: Option<String>,
    sends: u64,
}

#[derive(Clone, Default)]
pub struct ConversationSession {
    chat_jid: Arc<Mutex<Option<String>>>,
    workers: Arc<Mutex<HashMap<String, WorkerSession>>>,
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

    /// The history turn id a worker's outgoing messages are stamped with.
    pub fn set_turn(&self, worker_id: &str, turn_id: Option<&str>) {
        let mut workers = self.workers.lock().unwrap();
        workers.entry(worker_id.to_string()).or_default().turn_id = turn_id.map(str::to_string);
    }

    pub fn turn(&self, worker_id: &str) -> Option<String> {
        let workers = self.workers.lock().unwrap();
        workers.get(worker_id).and_then(|w| w.turn_id.clone())
    }

    /// Called after a message or reaction actually reaches the provider.
    pub fn record_send(&self, worker_id: &str) {
        let mut workers = self.workers.lock().unwrap();
        workers.entry(worker_id.to_string()).or_default().sends += 1;
    }

    /// Monotonic count; snapshot it before a turn and compare after.
    pub fn sends(&self, worker_id: &str) -> u64 {
        let workers = self.workers.lock().unwrap();
        workers.get(worker_id).map_or(0, |w| w.sends)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracks_sends_across_clones() {
        let session = ConversationSession::new();
        let snapshot = session.sends(FOREGROUND);

        // The MCP server holds its own clone of the same session.
        let mcp_side = session.clone();
        mcp_side.record_send(FOREGROUND);
        mcp_side.record_send(FOREGROUND);

        assert_eq!(session.sends(FOREGROUND) - snapshot, 2);
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

    #[test]
    fn test_workers_do_not_interfere() {
        let session = ConversationSession::new();
        session.set_turn(FOREGROUND, Some("turn_fg"));
        session.set_turn("schedule:1", Some("turn_sch"));
        let foreground_before = session.sends(FOREGROUND);

        session.record_send("schedule:1");
        session.set_turn("phoenix", Some("turn_phx"));
        session.set_turn("phoenix", None);

        assert_eq!(session.sends("schedule:1"), 1);
        assert_eq!(session.sends(FOREGROUND), foreground_before);
        assert_eq!(session.turn(FOREGROUND).as_deref(), Some("turn_fg"));
        assert_eq!(session.turn("schedule:1").as_deref(), Some("turn_sch"));
        assert_eq!(session.turn("phoenix"), None);
    }
}
