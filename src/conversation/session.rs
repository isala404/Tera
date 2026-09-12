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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct WorkerSession {
    turn_id: Option<String>,
    sends: u64,
}

#[derive(Clone, Default)]
pub struct ConversationSession {
    sends: Arc<AtomicU64>,
    chat_jid: Arc<Mutex<Option<String>>>,
    turn_id: Arc<Mutex<Option<String>>>,
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

    pub fn set_turn(&self, turn_id: Option<&str>) {
        *self.turn_id.lock().unwrap() = turn_id.map(str::to_string);
        self.set_turn_for("foreground", turn_id);
    }

    pub fn set_turn_for(&self, worker_id: &str, turn_id: Option<&str>) {
        let mut workers = self.workers.lock().unwrap();
        let session = workers.entry(worker_id.to_string()).or_default();
        session.turn_id = turn_id.map(str::to_string);
        if worker_id == "foreground" {
            *self.turn_id.lock().unwrap() = turn_id.map(str::to_string);
        }
    }

    pub fn turn(&self) -> Option<String> {
        self.turn_for(None)
    }

    pub fn turn_for(&self, worker_id: Option<&str>) -> Option<String> {
        if let Some(id) = worker_id {
            let workers = self.workers.lock().unwrap();
            if let Some(w) = workers.get(id) {
                return w.turn_id.clone();
            }
            return None;
        }
        self.turn_id.lock().unwrap().clone()
    }

    /// Called after a `send_message` tool call actually reaches the provider.
    pub fn record_send(&self) {
        self.record_send_for(None);
    }

    pub fn record_send_for(&self, worker_id: Option<&str>) {
        if let Some(id) = worker_id {
            let mut workers = self.workers.lock().unwrap();
            let session = workers.entry(id.to_string()).or_default();
            session.sends += 1;
            if id == "foreground" {
                self.sends.fetch_add(1, Ordering::SeqCst);
            }
        } else {
            self.sends.fetch_add(1, Ordering::SeqCst);
            let mut workers = self.workers.lock().unwrap();
            let session = workers.entry("foreground".to_string()).or_default();
            session.sends += 1;
        }
    }

    /// Monotonic count; snapshot it before a turn and compare after.
    pub fn count(&self) -> u64 {
        self.count_for(None)
    }

    pub fn count_for(&self, worker_id: Option<&str>) -> u64 {
        if let Some(id) = worker_id {
            let workers = self.workers.lock().unwrap();
            workers.get(id).map(|w| w.sends).unwrap_or(0)
        } else {
            self.sends.load(Ordering::SeqCst)
        }
    }

    pub fn sends_since(&self, snapshot: u64) -> u64 {
        self.sends_since_for(None, snapshot)
    }

    pub fn sends_since_for(&self, worker_id: Option<&str>, snapshot: u64) -> u64 {
        self.count_for(worker_id).saturating_sub(snapshot)
    }

    pub fn clear_worker(&self, worker_id: &str) {
        let mut workers = self.workers.lock().unwrap();
        if let Some(session) = workers.get_mut(worker_id) {
            session.turn_id = None;
        }
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

    #[test]
    fn test_worker_scoping_prevents_interference() {
        let session = ConversationSession::new();
        session.set_turn_for("foreground", Some("turn_fg"));
        let fg_snapshot = session.count_for(Some("foreground"));

        session.set_turn_for("schedule_1", Some("turn_sch"));
        let sch_snapshot = session.count_for(Some("schedule_1"));

        session.set_turn_for("phoenix", Some("turn_phx"));

        assert_eq!(
            session.turn_for(Some("foreground")),
            Some("turn_fg".to_string())
        );
        assert_eq!(
            session.turn_for(Some("schedule_1")),
            Some("turn_sch".to_string())
        );
        assert_eq!(
            session.turn_for(Some("phoenix")),
            Some("turn_phx".to_string())
        );

        // Schedule worker sends a message via MCP
        session.record_send_for(Some("schedule_1"));
        assert_eq!(session.sends_since_for(Some("schedule_1"), sch_snapshot), 1);
        assert_eq!(session.sends_since_for(Some("foreground"), fg_snapshot), 0);

        // Phoenix completes and clears its turn; foreground and schedule are unaffected
        session.clear_worker("phoenix");
        assert_eq!(
            session.turn_for(Some("foreground")),
            Some("turn_fg".to_string())
        );
        assert_eq!(
            session.turn_for(Some("schedule_1")),
            Some("turn_sch".to_string())
        );
    }
}
