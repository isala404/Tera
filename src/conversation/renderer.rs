use crate::history::db::ConversationEvent;
use chrono::{DateTime, Local, TimeZone};

pub struct InputRenderer;

impl InputRenderer {
    pub fn render_burst(events: &[ConversationEvent]) -> String {
        Self::render_events(events, None)
    }

    /// Render history for a fresh or recovering thread. Each message carries its
    /// full local timestamp and speaker so a relative request survives thread
    /// rotation with the temporal context that produced it.
    pub fn render_history(events: &[ConversationEvent]) -> String {
        Self::render_events(events, Some("Recent conversation context from canonical history"))
    }

    fn render_events(events: &[ConversationEvent], heading: Option<&str>) -> String {
        let mut rendered = String::new();

        // The agent has no clock of its own. Without the date and UTC offset it
        // cannot convert "in five minutes" into a timestamp, and scheduling
        // silently lands in the past, which fires every task immediately.
        rendered.push_str(&format!(
            "[Current time: {}]\n\n",
            Local::now().format("%Y-%m-%d %H:%M:%S %:z (%Z)")
        ));

        if let Some(heading) = heading {
            rendered.push_str(heading);
            rendered.push_str("\n\n");
        }

        for event in events {
            let dt: DateTime<Local> = Local.timestamp_millis_opt(event.occurred_at_ms).unwrap();
            let t_str = dt.format("%Y-%m-%d %H:%M:%S %:z").to_string();
            let speaker = match event.actor.as_str() {
                "assistant" => "Assistant",
                "user" => "User",
                other => other,
            };

            // The id is the handle for `react` and for send_message's reply_to.
            // Without it in the transcript the agent can see that something was
            // replied to but has no way to name anything itself.
            rendered.push_str(&format!("[{}] {} {}", t_str, speaker, event.id));

            if let Some(ref reply_to) = event.reply_to_id {
                rendered.push_str(&format!(" (replying to {})", reply_to));
            }
            rendered.push_str(":\n");

            if let Some(ref text) = event.text {
                rendered.push_str(text);
                rendered.push('\n');
            }

            for att in &event.attachments {
                rendered.push_str(&format!(
                    "[Attachment {}: {} ({})]\n",
                    att.media_type, att.relative_path, att.original_name.as_deref().unwrap_or("unknown")
                ));
            }

            rendered.push('\n');
        }

        rendered.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_burst() {
        let ev = ConversationEvent {
            seq: None,
            id: "m_test".to_string(),
            occurred_at_ms: 1700000000000,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: Some("Test query".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        };

        let rendered = InputRenderer::render_burst(&[ev]);
        assert!(rendered.contains("User m_test:"), "{rendered}");
        assert!(rendered.contains("Test query"));
    }

    #[test]
    fn test_history_renders_full_timestamps_and_speakers() {
        let user_at = 1_700_000_000_000;
        let assistant_at = user_at + 1_000;
        let events = [
            ConversationEvent {
                seq: None,
                id: "m_user".to_string(),
                occurred_at_ms: user_at,
                kind: "message".to_string(),
                actor: "user".to_string(),
                text: Some("What happened yesterday?".to_string()),
                reply_to_id: None,
                turn_id: None,
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            },
            ConversationEvent {
                seq: None,
                id: "m_assistant".to_string(),
                occurred_at_ms: assistant_at,
                kind: "message".to_string(),
                actor: "assistant".to_string(),
                text: Some("You asked about yesterday.".to_string()),
                reply_to_id: None,
                turn_id: None,
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            },
        ];

        let rendered = InputRenderer::render_history(&events);
        let user_stamp = Local
            .timestamp_millis_opt(user_at)
            .single()
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S %:z")
            .to_string();
        let assistant_stamp = Local
            .timestamp_millis_opt(assistant_at)
            .single()
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S %:z")
            .to_string();
        assert!(rendered.contains(&format!("[{user_stamp}] User")));
        assert!(rendered.contains(&format!("[{assistant_stamp}] Assistant")));
        assert!(rendered.contains("What happened yesterday?"));
        assert!(rendered.contains("You asked about yesterday."));
    }
}
