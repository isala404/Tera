use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    pub provider_msg_id: String,
    pub sender: String,
    pub text: Option<String>,
    pub timestamp_ms: i64,
    pub reply_to_provider_msg_id: Option<String>,
    pub media_attachment: Option<InboundMedia>,
    pub media_error: Option<String>,
    /// Chat this message belongs to, without a device suffix. Replies and
    /// reactions must address this, not the sending device.
    pub chat_jid: String,
    /// Sent by the account this daemon is paired to, from any of its devices.
    ///
    /// Not the same as the SDK's `is_from_me`, which is only true for messages
    /// *this* device sent, a message typed on the owner's phone arrives with
    /// `is_from_me = false` and a device suffix on the JID.
    pub from_own_account: bool,
    pub is_group: bool,
}

impl InboundMessage {
    /// Whether there is anything here for the assistant to answer.
    ///
    /// WhatsApp delivers protocol traffic and history sync replays through the
    /// same callback as real chat, and those arrive carrying neither text nor an
    /// attachment. Starting a turn for one gives the model nothing to work with,
    /// and at pairing a burst of them opened a turn on the paired account's own
    /// chat that the owner's first real message was then steered into.
    pub fn is_answerable(&self) -> bool {
        self.media_attachment.is_some()
            || self.media_error.is_some()
            || self
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
    }
}

/// The owner's live chat state. It is deliberately smaller than WhatsApp's
/// event: the turn engine only needs to know whether more input is still coming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InboundPresenceKind {
    Typing,
    RecordingAudio,
    Paused,
}

#[derive(Debug, Clone)]
pub struct InboundPresence {
    pub sender: String,
    pub kind: InboundPresenceKind,
    pub from_own_account: bool,
    pub is_group: bool,
}

/// Media that arrived with a message, already fetched and decrypted.
///
/// The bytes are carried rather than a provider handle: WhatsApp media URLs are
/// short-lived and single-use, so a download that is deferred is a download that
/// fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMedia {
    pub media_type: String, // "image", "video", "audio", "document", "sticker"
    pub mime_type: String,
    pub filename: String,
    pub data: Vec<u8>,
}

/// Everything needed to point at an existing message: to react to it, or to
/// quote it in a reply.
///
/// A provider message id alone is not enough. WhatsApp keys a message by chat,
/// sender-side and id together, and renders a quote from a copy of the original
/// carried in the reply itself. Passing only the id meant guessing the rest, and
/// a wrong guess is accepted by the server and then silently dropped.
#[derive(Debug, Clone)]
pub struct MessageRef {
    pub provider_msg_id: String,
    /// Chat the message lives in, without any device suffix.
    pub chat_jid: String,
    /// Whether the target message was sent by this account.
    pub from_me: bool,
    /// The message's own text, so a quote of it can render on the recipient's
    /// phone. `None` where there is no text to show, such as a bare attachment.
    pub text: Option<String>,
}

#[async_trait]
pub trait Transport: Send + Sync {
    async fn send_text(
        &self,
        recipient: &str,
        text: &str,
        reply_to: Option<&MessageRef>,
    ) -> Result<String>;
    async fn send_media(
        &self,
        recipient: &str,
        media_type: &str,
        file_path: &Path,
        caption: Option<&str>,
        reply_to: Option<&MessageRef>,
    ) -> Result<String>;
    async fn send_reaction(&self, recipient: &str, target: &MessageRef, emoji: &str) -> Result<()>;
    async fn set_typing_status(&self, recipient: &str, typing: bool) -> Result<()>;
}

pub mod owner;
pub mod whatsapp;
pub use owner::{OwnerPolicy, Verdict};
pub use whatsapp::{MockTransport, WhatsAppWebTransport};

#[cfg(test)]
mod tests {
    use super::*;

    fn message(text: Option<&str>, media: Option<InboundMedia>) -> InboundMessage {
        InboundMessage {
            provider_msg_id: "wa_1".to_string(),
            sender: "owner:26@s.whatsapp.net".to_string(),
            text: text.map(str::to_string),
            timestamp_ms: 0,
            reply_to_provider_msg_id: None,
            media_attachment: media,
            media_error: None,
            chat_jid: "owner@s.whatsapp.net".to_string(),
            from_own_account: true,
            is_group: false,
        }
    }

    fn media() -> InboundMedia {
        InboundMedia {
            media_type: "image".to_string(),
            mime_type: "image/jpeg".to_string(),
            filename: "photo.jpg".to_string(),
            data: vec![1, 2, 3],
        }
    }

    /// History sync replays arrive here with nothing in them. One of those
    /// opened the turn that swallowed the owner's first message after a
    /// re-pairing, so this is the guard that keeps them out.
    #[test]
    fn test_a_message_with_no_text_and_no_media_is_not_answerable() {
        assert!(!message(None, None).is_answerable());
        assert!(!message(Some(""), None).is_answerable());
        assert!(!message(Some("   \n"), None).is_answerable());
    }

    #[test]
    fn test_text_or_media_makes_a_message_answerable() {
        assert!(message(Some("hi"), None).is_answerable());
        // A bare photo carries no text and still has to be answered.
        assert!(message(None, Some(media())).is_answerable());

        let mut failed = message(None, None);
        failed.media_error = Some("failed to download".to_string());
        assert!(failed.is_answerable());
    }
}
