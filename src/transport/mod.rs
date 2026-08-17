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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundReaction {
    pub provider_msg_id: String,
    pub sender: String,
    pub target_provider_msg_id: String,
    pub emoji: String,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone)]
pub enum InboundEvent {
    Message(InboundMessage),
    Reaction(InboundReaction),
}

/// Everything needed to address a reaction at an existing message.
///
/// A provider message id alone is not enough: WhatsApp keys a reaction by chat,
/// sender-side and id together. Passing only the id meant guessing the other
/// two, and a wrong guess is accepted by the server and then silently dropped.
#[derive(Debug, Clone)]
pub struct ReactionTarget {
    pub provider_msg_id: String,
    /// Chat the message lives in, without any device suffix.
    pub chat_jid: String,
    /// Whether the target message was sent by this account.
    pub from_me: bool,
}

#[async_trait]
pub trait Transport: Send + Sync {
    async fn send_text(&self, recipient: &str, text: &str, reply_to_provider_id: Option<&str>) -> Result<String>;
    async fn send_media(
        &self,
        recipient: &str,
        media_type: &str,
        file_path: &Path,
        caption: Option<&str>,
        reply_to_provider_id: Option<&str>,
    ) -> Result<String>;
    async fn send_reaction(&self, recipient: &str, target: &ReactionTarget, emoji: &str) -> Result<()>;
    async fn set_typing_status(&self, recipient: &str, typing: bool) -> Result<()>;
}

pub mod owner;
pub mod whatsapp;
pub use owner::{OwnerPolicy, Verdict};
pub use whatsapp::{MockTransport, WhatsAppWebTransport};
