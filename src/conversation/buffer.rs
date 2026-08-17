use crate::history::db::ConversationEvent;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct MessageBurst {
    pub turn_id: String,
    pub events: Vec<ConversationEvent>,
    pub created_at: Instant,
    pub last_updated_at: Instant,
}

impl MessageBurst {
    pub fn new(turn_id: String, event: ConversationEvent) -> Self {
        let now = Instant::now();
        Self {
            turn_id,
            events: vec![event],
            created_at: now,
            last_updated_at: now,
        }
    }

    pub fn push(&mut self, event: ConversationEvent) {
        self.events.push(event);
        self.last_updated_at = Instant::now();
    }

    /// How long to keep waiting before starting the turn.
    ///
    /// The quiet period restarts on every message, so a user who keeps typing
    /// would otherwise never get an answer. `max_wait` caps the total from the
    /// first message (PLAN.md section 13.1).
    pub fn remaining_wait(&self, quiet: Duration, max_wait: Duration) -> Duration {
        let since_last = self.last_updated_at.elapsed();
        let quiet_left = quiet.saturating_sub(since_last);
        let max_left = max_wait.saturating_sub(self.created_at.elapsed());
        quiet_left.min(max_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_burst_buffering() {
        let ev1 = ConversationEvent {
            seq: None,
            id: "m_1".to_string(),
            occurred_at_ms: 1000,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: Some("Hello".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        };

        let mut burst = MessageBurst::new("turn_1".to_string(), ev1);
        assert_eq!(burst.events.len(), 1);

        let ev2 = ConversationEvent {
            seq: None,
            id: "m_2".to_string(),
            occurred_at_ms: 1005,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: Some("World".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        };

        burst.push(ev2);
        assert_eq!(burst.events.len(), 2);
        assert_eq!(burst.turn_id, "turn_1");
    }

    #[test]
    fn test_quiet_period_is_what_normally_bounds_the_wait() {
        let burst = MessageBurst::new("turn_1".to_string(), event("m_1"));
        let remaining = burst.remaining_wait(Duration::from_millis(2500), Duration::from_secs(8));
        assert!(remaining > Duration::from_millis(2000));
        assert!(remaining <= Duration::from_millis(2500));
    }

    /// A user who keeps typing restarts the quiet period every time. Without a
    /// ceiling on the total wait, they never get an answer.
    #[test]
    fn test_a_long_burst_is_capped_by_the_maximum_wait() {
        let burst = MessageBurst::new("turn_1".to_string(), event("m_1"));
        // Max wait already elapsed: nothing left to wait for, whatever the quiet
        // period says.
        let remaining = burst.remaining_wait(Duration::from_secs(3), Duration::ZERO);
        assert_eq!(remaining, Duration::ZERO);
    }

    fn event(id: &str) -> ConversationEvent {
        ConversationEvent {
            seq: None,
            id: id.to_string(),
            occurred_at_ms: 1000,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: Some("hi".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        }
    }
}
