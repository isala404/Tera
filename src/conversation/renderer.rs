use crate::history::db::ConversationEvent;
use chrono::{DateTime, Local, TimeZone};

pub struct InputRenderer;

impl InputRenderer {
    pub fn render_burst(events: &[ConversationEvent]) -> String {
        let mut rendered = String::new();

        // The agent has no clock of its own. Without the date and UTC offset it
        // cannot convert "in five minutes" into a timestamp, and scheduling
        // silently lands in the past, which fires every task immediately.
        rendered.push_str(&format!(
            "[Current time: {}]\n\n",
            Local::now().format("%Y-%m-%d %H:%M:%S %:z (%Z)")
        ));

        for event in events {
            let dt: DateTime<Local> = Local.timestamp_millis_opt(event.occurred_at_ms).unwrap();
            let t_str = dt.format("%H:%M:%S").to_string();

            rendered.push_str(&format!("[{}] User", t_str));

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
        assert!(rendered.contains("User:"));
        assert!(rendered.contains("Test query"));
    }
}
