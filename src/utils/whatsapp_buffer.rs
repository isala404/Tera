#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use wacore::types::message::MessageInfo;
use waproto::whatsapp as wa;

const DEBOUNCE_DURATION: Duration = Duration::from_secs(3);
const TYPING_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct BufferedMessage {
    pub message: wa::Message,
    pub info: MessageInfo,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypingState {
    Idle,
    Typing,
}

pub struct ChatBuffer {
    pub messages: Vec<BufferedMessage>,
    pub typing_state: TypingState,
    pub last_activity: Instant,
}

impl ChatBuffer {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            typing_state: TypingState::Idle,
            last_activity: Instant::now(),
        }
    }

    pub fn is_ready_to_flush(&self) -> bool {
        if self.messages.is_empty() {
            return false;
        }

        let now = Instant::now();
        let typing_timed_out = now.duration_since(self.last_activity) > TYPING_TIMEOUT;
        let debounce_passed = now.duration_since(self.last_activity) >= DEBOUNCE_DURATION;

        match self.typing_state {
            TypingState::Idle => debounce_passed,
            TypingState::Typing => typing_timed_out,
        }
    }

    pub fn update_message(&mut self, message_id: &str, new_message: wa::Message) -> bool {
        for buffered in &mut self.messages {
            if buffered.info.id == message_id {
                buffered.message = new_message;
                return true;
            }
        }
        false
    }
}

#[derive(Clone)]
pub struct BufferManager {
    buffers: Arc<RwLock<HashMap<String, ChatBuffer>>>,
}

impl BufferManager {
    pub fn new() -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add(&self, chat_jid: String, message: wa::Message, info: MessageInfo) {
        let mut buffers = self.buffers.write().await;
        let buffer = buffers.entry(chat_jid).or_insert_with(ChatBuffer::new);
        buffer.messages.push(BufferedMessage { message, info });
        buffer.last_activity = Instant::now();
    }

    pub async fn update_typing(&self, chat_jid: &str, state: TypingState) {
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(chat_jid) {
            buffer.typing_state = state;
            buffer.last_activity = Instant::now();
        }
    }

    pub async fn get_ready_chats(&self) -> Vec<String> {
        let buffers = self.buffers.read().await;
        buffers
            .iter()
            .filter(|(_, buffer)| buffer.is_ready_to_flush())
            .map(|(jid, _)| jid.clone())
            .collect()
    }

    pub async fn flush(&self, chat_jid: &str) -> Vec<BufferedMessage> {
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(chat_jid) {
            let messages = std::mem::take(&mut buffer.messages);
            buffer.typing_state = TypingState::Idle;
            messages
        } else {
            Vec::new()
        }
    }

    pub async fn update_buffered_message(
        &self,
        chat_jid: &str,
        message_id: &str,
        new_message: wa::Message,
    ) -> bool {
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(chat_jid) {
            buffer.update_message(message_id, new_message)
        } else {
            false
        }
    }

    pub async fn has_pending(&self, chat_jid: &str) -> bool {
        let buffers = self.buffers.read().await;
        buffers
            .get(chat_jid)
            .map(|b| !b.messages.is_empty())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferManager, ChatBuffer, TypingState};
    use std::time::{Duration, Instant};
    use wacore::types::message::MessageInfo;
    use waproto::whatsapp as wa;

    #[test]
    fn test_chat_buffer_not_ready_when_empty() {
        let buffer = ChatBuffer::new();
        assert!(!buffer.is_ready_to_flush());
    }

    #[test]
    fn test_chat_buffer_ready_when_idle_and_debounced() {
        let mut buffer = ChatBuffer::new();
        buffer.messages.push(super::BufferedMessage {
            message: wa::Message::default(),
            info: MessageInfo::default(),
        });
        buffer.typing_state = TypingState::Idle;
        buffer.last_activity = Instant::now() - Duration::from_secs(4);
        assert!(buffer.is_ready_to_flush());
    }

    #[tokio::test]
    async fn test_buffer_manager_add_and_flush() {
        let manager = BufferManager::new();
        manager
            .add(
                "chat-1".to_string(),
                wa::Message::default(),
                MessageInfo::default(),
            )
            .await;

        assert!(manager.has_pending("chat-1").await);
        let flushed = manager.flush("chat-1").await;
        assert_eq!(flushed.len(), 1);
        assert!(!manager.has_pending("chat-1").await);
    }
}
