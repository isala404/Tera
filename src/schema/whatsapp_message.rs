use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use uuid::Uuid;

#[forge::forge_enum]
pub enum MessageDirection {
    In,
    Out,
}

#[forge::forge_enum]
pub enum MessageStatus {
    PendingAgent,
    SentAgent,
    PendingGateway,
    SentGateway,
    FailedGateway,
}

#[forge::model]
#[allow(dead_code)]
pub struct WhatsappMessage {
    pub id: Uuid,
    pub chat_id: String,
    pub direction: MessageDirection,
    pub agent_id: Option<Uuid>,
    pub status: MessageStatus,
    pub content_text: Option<String>,
    pub media: Option<Value>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl WhatsappMessage {
    #[allow(dead_code)]
    pub fn new(chat_id: String, direction: MessageDirection) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            chat_id,
            direction,
            agent_id: None,
            status: MessageStatus::PendingAgent,
            content_text: None,
            media: None,
            embedding: None,
            metadata: json!({}),
            created_at: now,
            updated_at: now,
        }
    }

    #[allow(dead_code)]
    pub fn with_text(mut self, text: String) -> Self {
        self.content_text = Some(text);
        self
    }

    #[allow(dead_code)]
    pub fn with_media(mut self, media: Value) -> Self {
        self.media = Some(media);
        self
    }

    #[allow(dead_code)]
    pub fn with_agent_id(mut self, agent_id: Uuid) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    #[allow(dead_code)]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}
