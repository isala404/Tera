use crate::transport::owner::jid_user;
use crate::transport::{InboundEvent, InboundMedia, InboundMessage, ReactionTarget, Transport};
use anyhow::{anyhow, Context, Result};
use qrcode::render::unicode;
use qrcode::QrCode;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use uuid::Uuid;
use whatsapp_rust::bot::MessageContext;
use whatsapp_rust::media;
use whatsapp_rust::wacore::download::MediaType;
use whatsapp_rust::upload::{UploadOptions, UploadResponse};
use whatsapp_rust::prelude::*;
use whatsapp_rust_sqlite_storage::SqliteStore;

/// Render the pairing payload as a scannable QR block on the terminal.
///
/// Printing the raw payload string was useless. WhatsApp's "link a device"
/// flow only accepts a camera scan, so the code has to be drawn.
///
/// Uses half-block characters so one text row carries two QR rows, keeping the
/// symbol square-ish and small enough to fit an 80-column terminal. Drawn dark-on-
/// light with an explicit quiet zone, because scanners need the light margin and
/// most terminals are dark-themed.
fn print_pairing_qr(payload: &str, timeout: std::time::Duration) {
    let rendered = QrCode::new(payload.as_bytes()).map(|code| {
        code.render::<unicode::Dense1x2>()
            .quiet_zone(true)
            .dark_color(unicode::Dense1x2::Light)
            .light_color(unicode::Dense1x2::Dark)
            .build()
    });

    println!();
    match rendered {
        Ok(qr) => {
            println!("  Scan to link this device (WhatsApp > Linked devices > Link a device)");
            println!("  This code expires in {}s\n", timeout.as_secs());
            println!("{qr}");
        }
        Err(e) => {
            // Never leave the operator stuck: the raw payload can still be fed
            // into any external QR generator.
            warn!("Could not render pairing QR code: {e}");
            println!("  Pairing payload (render it yourself): {payload}");
        }
    }
    println!();
}

/// Best-effort MIME type for an outgoing file, from its extension.
///
/// WhatsApp clients decide how to render an attachment from this, so a video
/// labelled application/octet-stream shows up as an unplayable file.
fn mime_for(media_type: &str, path: &Path) -> String {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "mp4" | "m4v" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "ogg" | "opus" => "audio/ogg; codecs=opus",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "wav" => "audio/wav",
        "pdf" => "application/pdf",
        _ => match media_type {
            "image" => "image/jpeg",
            "video" => "video/mp4",
            "audio" => "audio/ogg; codecs=opus",
            _ => "application/octet-stream",
        },
    }
    .to_string()
}

/// Name an attachment after the provider message id so the asset directory is
/// self-describing, using the mime subtype as the extension when it looks sane.
fn media_filename(msg_id: &str, mime: &str, default_ext: &str) -> String {
    let ext = mime
        .split('/')
        .nth(1)
        .map(|s| s.split(';').next().unwrap_or(s))
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or(default_ext);
    format!("{msg_id}.{ext}")
}

pub struct WhatsAppWebTransport {
    session_db_path: PathBuf,
    client_handle: Arc<Mutex<Option<Arc<Client>>>>,
}

impl WhatsAppWebTransport {
    pub fn new(session_db_path: PathBuf) -> Self {
        Self {
            session_db_path,
            client_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Whether a message came from the account this daemon is paired to.
    ///
    /// `is_from_me` alone is not enough: it is only true for messages this
    /// linked device sent, so the owner typing on their own phone arrives as
    /// `is_from_me = false` with a device-suffixed JID like `...:26@lid`.
    /// Comparing the sender's user part against our own LID and phone-number
    /// JIDs recognises the owner from any of their devices.
    fn is_own_account(ctx: &MessageContext) -> bool {
        if ctx.info.source.is_from_me {
            return true;
        }

        let sender = ctx.info.source.sender.to_string();
        let sender_user = jid_user(&sender);

        [ctx.client.lid(), ctx.client.pn()]
            .iter()
            .flatten()
            .any(|own| jid_user(&own.to_string()) == sender_user)
    }

    /// Download and decrypt any media attached to an inbound message.
    ///
    /// Done eagerly, inside the message handler: WhatsApp media references are
    /// short-lived, so a message parked in a burst buffer for a few seconds can
    /// no longer be fetched by the time the turn runs. Failure degrades to no
    /// attachment, the text still gets through, and the caller can see that the
    /// media is missing rather than silently answering blind.
    async fn fetch_media(ctx: &MessageContext) -> Option<InboundMedia> {
        let base = ctx.message.get_base_message();

        macro_rules! try_download {
            ($field:expr, $kind:literal, $default_ext:literal) => {
                if let Some(media) = $field.as_option() {
                    let mime = media.mimetype.clone().unwrap_or_default();
                    return match ctx.client.download(media).await {
                        Ok(data) => {
                            info!("Downloaded {} attachment ({} bytes)", $kind, data.len());
                            Some(InboundMedia {
                                media_type: $kind.to_string(),
                                filename: media_filename(&ctx.info.id, &mime, $default_ext),
                                mime_type: mime,
                                data,
                            })
                        }
                        Err(e) => {
                            warn!("Failed to download {} attachment: {:?}", $kind, e);
                            None
                        }
                    };
                }
            };
        }

        try_download!(base.image_message, "image", "jpg");
        try_download!(base.video_message, "video", "mp4");
        try_download!(base.audio_message, "audio", "ogg");
        try_download!(base.document_message, "document", "bin");
        try_download!(base.sticker_message, "sticker", "webp");

        None
    }

    async fn upload(
        &self,
        client: &Arc<Client>,
        data: Vec<u8>,
        media_type: MediaType,
    ) -> Result<UploadResponse> {
        client
            .upload(data, media_type, UploadOptions::default())
            .await
            .map_err(|e| anyhow!("WhatsApp media upload failed: {:?}", e))
    }

    /// The connected client plus a parsed recipient, or an error explaining
    /// which of the two is missing.
    fn client_for(&self, recipient: &str) -> Result<(Arc<Client>, Jid)> {
        let client = self
            .client_handle
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow!("WhatsApp client is not connected yet"))?;
        let jid: Jid = recipient
            .parse()
            .map_err(|e| anyhow!("Invalid recipient JID '{}': {:?}", recipient, e))?;
        Ok((client, jid))
    }

    pub async fn start_bot<F>(&self, inbound_callback: F) -> Result<()>
    where
        F: Fn(InboundEvent) + Send + Sync + 'static,
    {
        let db_str = self.session_db_path.to_string_lossy().to_string();
        info!("Starting whatsapp-rust session at {}", db_str);

        let store = SqliteStore::new(&db_str)
            .await
            .map_err(|e| anyhow!("Failed to initialize whatsapp-rust SqliteStore: {:?}", e))?;

        let callback_arc = Arc::new(inbound_callback);

        let bot = Bot::builder()
            .with_backend(store)
            .on_qr_code(|code, timeout| async move {
                print_pairing_qr(&code, timeout);
            })
            // Presence has to wait for the socket. Setting it right after
            // `build()` raced the connection and always failed with NotConnected;
            // this also re-announces after each reconnect.
            .on_connected(|client: Arc<Client>| async move {
                info!(
                    "Connected as lid={:?} pn={:?}",
                    client.lid().map(|j| j.to_string()),
                    client.pn().map(|j| j.to_string())
                );
                if let Err(e) = client.presence().set_available().await {
                    warn!("Could not set WhatsApp presence to available: {:?}", e);
                }
            })
            .on_message(move |ctx| {
                let cb = callback_arc.clone();
                async move {
                    let sender = ctx.info.source.sender.to_string();
                    let text = ctx
                        .message
                        .text_content()
                        .or_else(|| ctx.message.get_caption())
                        .map(|s| s.to_string());

                    let media_attachment = Self::fetch_media(&ctx).await;
                    let timestamp_ms = ctx.info.timestamp.timestamp_millis();

                    let msg = InboundMessage {
                        provider_msg_id: ctx.info.id.clone(),
                        sender,
                        text,
                        timestamp_ms,
                        reply_to_provider_msg_id: None,
                        media_attachment,
                        chat_jid: ctx.info.source.chat.to_non_ad_string(),
                        from_own_account: Self::is_own_account(&ctx),
                        is_group: ctx.info.source.is_group,
                    };

                    cb(InboundEvent::Message(msg));
                }
            })
            .build()
            .await
            .map_err(|e| anyhow!("Failed to build whatsapp-rust bot: {:?}", e))?;

        // Save active Client handle for outbound message dispatching
        let client = bot.client();
        {
            let mut lock = self.client_handle.lock().unwrap();
            *lock = Some(client.clone());
        }
        info!("whatsapp-rust session started; waiting for connection...");
        bot.run().await;
        Ok(())
    }
}

#[async_trait]
impl Transport for WhatsAppWebTransport {
    async fn send_text(&self, recipient: &str, text: &str, _reply_to_provider_id: Option<&str>) -> Result<String> {
        info!("whatsapp-rust send_text to {}: {}", recipient, text);
        let client_opt = {
            let lock = self.client_handle.lock().unwrap();
            lock.clone()
        };

        // Never invent a message id on failure: the caller records the returned
        // id in canonical history, so a fabricated one writes a message the user
        // never received into the permanent record (PLAN.md section 55).
        let client = client_opt.ok_or_else(|| anyhow!("WhatsApp client is not connected yet"))?;
        let jid: Jid = recipient
            .parse()
            .map_err(|e| anyhow!("Invalid recipient JID '{}': {:?}", recipient, e))?;

        let send_res = client
            .send_text(jid, text)
            .await
            .map_err(|e| anyhow!("WhatsApp send_text to {} failed: {:?}", recipient, e))?;

        info!("Sent WhatsApp message to {}: {:?}", recipient, send_res.message_id);
        Ok(send_res.message_id)
    }

    async fn send_media(
        &self,
        recipient: &str,
        media_type: &str,
        file_path: &Path,
        caption: Option<&str>,
        reply_to_provider_id: Option<&str>,
    ) -> Result<String> {
        let (client, jid) = self.client_for(recipient)?;

        let data = std::fs::read(file_path)
            .with_context(|| format!("Cannot read media file {}", file_path.display()))?;
        let mime = mime_for(media_type, file_path);
        info!(
            "Uploading {} ({} bytes, {}) from {}",
            media_type,
            data.len(),
            mime,
            file_path.display()
        );

        let (wa_media_type, message) = match media_type {
            "image" => {
                let upload = self.upload(&client, data, MediaType::Image).await?;
                (
                    MediaType::Image,
                    media::image_message(
                        upload,
                        media::ImageOptions {
                            caption: caption.map(str::to_string),
                            mimetype: Some(mime),
                            ..Default::default()
                        },
                    ),
                )
            }
            "video" => {
                let upload = self.upload(&client, data, MediaType::Video).await?;
                (
                    MediaType::Video,
                    media::video_message(
                        upload,
                        media::VideoOptions {
                            caption: caption.map(str::to_string),
                            mimetype: Some(mime),
                            ..Default::default()
                        },
                    ),
                )
            }
            "audio" => {
                let upload = self.upload(&client, data, MediaType::Audio).await?;
                (
                    MediaType::Audio,
                    media::audio_message(
                        upload,
                        media::AudioOptions {
                            mimetype: Some(mime),
                            ..Default::default()
                        },
                    ),
                )
            }
            // Anything else rides as a document, which is what WhatsApp does for
            // arbitrary files anyway.
            _ => {
                let upload = self.upload(&client, data, MediaType::Document).await?;
                (
                    MediaType::Document,
                    media::document_message(
                        upload,
                        media::DocumentOptions {
                            mimetype: Some(mime),
                            file_name: file_path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string()),
                            caption: caption.map(str::to_string),
                            ..Default::default()
                        },
                    ),
                )
            }
        };
        let _ = (wa_media_type, reply_to_provider_id);

        let send_res = client
            .send_message(jid, message)
            .await
            .map_err(|e| anyhow!("WhatsApp send_media to {} failed: {:?}", recipient, e))?;

        info!(
            "Sent {} to {}: {:?}",
            media_type, recipient, send_res.message_id
        );
        Ok(send_res.message_id)
    }

    async fn send_reaction(
        &self,
        recipient: &str,
        target: &ReactionTarget,
        emoji: &str,
    ) -> Result<()> {
        let (client, jid) = self.client_for(recipient)?;

        // The key must identify the message exactly as WhatsApp stored it: the
        // chat JID without a device suffix, and whether the target was sent by
        // this account. Reacting with a device-suffixed JID is accepted locally
        // and then silently ignored, which is why the thumbs-up never appeared.
        let chat_jid: Jid = target
            .chat_jid
            .parse()
            .map_err(|e| anyhow!("Invalid chat JID '{}': {:?}", target.chat_jid, e))?;

        let key = wa::MessageKey {
            remote_jid: Some(chat_jid.to_non_ad_string()),
            from_me: Some(target.from_me),
            id: Some(target.provider_msg_id.clone()),
            participant: None,
        };

        client
            .send_reaction(jid, key, emoji)
            .await
            .map_err(|e| anyhow!("WhatsApp reaction failed: {:?}", e))?;

        info!(
            "Reacted {} on message {} in chat {} (from_me={})",
            emoji, target.provider_msg_id, target.chat_jid, target.from_me
        );
        Ok(())
    }

    async fn set_typing_status(&self, recipient: &str, typing: bool) -> Result<()> {
        let (client, jid) = self.client_for(recipient)?;
        let chatstate = client.chatstate();

        let result = if typing {
            chatstate.send_composing(&jid).await
        } else {
            chatstate.send_paused(&jid).await
        };

        result.map_err(|e| anyhow!("WhatsApp chat state update failed: {:?}", e))
    }
}

/// Recipient, text, and the provider id it replied to.
type SentMessage = (String, String, Option<String>);

#[derive(Default)]
pub struct MockTransport {
    pub sent_messages: Arc<Mutex<Vec<SentMessage>>>,
    pub sent_reactions: Arc<Mutex<Vec<(String, String, String)>>>,
    pub typing_states: Arc<Mutex<Vec<(String, bool)>>>,
    pub inbound_queue: Arc<Mutex<VecDeque<InboundEvent>>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_inbound(&self, event: InboundEvent) {
        self.inbound_queue.lock().unwrap().push_back(event);
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send_text(&self, recipient: &str, text: &str, reply_to_provider_id: Option<&str>) -> Result<String> {
        let msg_id = format!("wamid.mock_{}", Uuid::new_v4().simple());
        self.sent_messages.lock().unwrap().push((
            recipient.to_string(),
            text.to_string(),
            reply_to_provider_id.map(|s| s.to_string()),
        ));
        Ok(msg_id)
    }

    async fn send_media(
        &self,
        recipient: &str,
        media_type: &str,
        file_path: &Path,
        caption: Option<&str>,
        reply_to_provider_id: Option<&str>,
    ) -> Result<String> {
        let msg_id = format!("wamid.mock_media_{}", Uuid::new_v4().simple());
        let cap_str = caption.unwrap_or("");
        let text = format!("[MockMedia:{}:{}] {}", media_type, file_path.display(), cap_str);
        self.sent_messages.lock().unwrap().push((
            recipient.to_string(),
            text,
            reply_to_provider_id.map(|s| s.to_string()),
        ));
        Ok(msg_id)
    }

    async fn send_reaction(&self, recipient: &str, target: &ReactionTarget, emoji: &str) -> Result<()> {
        self.sent_reactions.lock().unwrap().push((
            recipient.to_string(),
            target.provider_msg_id.clone(),
            emoji.to_string(),
        ));
        Ok(())
    }

    async fn set_typing_status(&self, recipient: &str, typing: bool) -> Result<()> {
        self.typing_states.lock().unwrap().push((recipient.to_string(), typing));
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use qrcode::Color;

    /// Reconstruct the module grid from the terminal art.
    ///
    /// `Dense1x2` packs two vertical modules into one character, and the renderer
    /// is configured to invert the colours so a full block is a LIGHT area and a
    /// space is a DARK module. That inversion is what makes the symbol scannable on
    /// a dark terminal, and it is also the easiest thing to get backwards, so the
    /// test reads the art the same way a camera would rather than trusting it.
    fn modules_from_art(art: &str) -> Vec<Vec<bool>> {
        let mut rows = Vec::new();
        for line in art.lines() {
            let mut top = Vec::new();
            let mut bottom = Vec::new();
            for ch in line.chars() {
                let (t, b) = match ch {
                    '\u{2588}' => (false, false), // full block: both halves light
                    '\u{2580}' => (false, true),  // upper half block: bottom is dark
                    '\u{2584}' => (true, false),  // lower half block: top is dark
                    ' ' => (true, true),           // blank: both halves dark
                    other => panic!("unexpected glyph {other:?} in the QR"),
                };
                top.push(t);
                bottom.push(b);
            }
            rows.push(top);
            rows.push(bottom);
        }
        rows
    }

    /// The pairing QR has to survive a phone camera, and "it rendered without an
    /// error" proves nothing about that. This checks the art is a faithful,
    /// correctly-oriented copy of the modules the encoder produced, with the quiet
    /// zone a scanner needs, narrow enough not to wrap.
    ///
    /// Verified once against a real decoder (OpenCV round-tripped the payload from
    /// this exact rendering); what can be asserted cheaply on every run is that the
    /// art still matches the grid.
    #[test]
    fn test_pairing_qr_is_a_faithful_scannable_rendering() {
        // A representative whatsapp-web pairing payload: 4 comma-separated parts.
        let payload = "2@Kx8vQZ1mN7pLcR4tYuIoP3aSdFgHjKlZxCvBnM6qWeRtYuIoP0aSdFgHjKlZxCvBnM2q,\
                       7ZkLpQwErTyUiOpAsDfGhJkLzXcVbNm4QwErTyUiOpAsDfGhJ=,\
                       Yh3RfVbGtYhNuJmIkOlPzAqWsXeDcRfVbGtYhNuJmIkOlP0=,1";
        let code = QrCode::new(payload.as_bytes()).unwrap();
        let width = code.width();
        let colors = code.to_colors();

        let art = code
            .render::<unicode::Dense1x2>()
            .quiet_zone(true)
            .dark_color(unicode::Dense1x2::Light)
            .light_color(unicode::Dense1x2::Dark)
            .build();

        // Must stay inside a standard terminal, or the symbol wraps and cannot scan.
        let widest = art.lines().map(|l| l.chars().count()).max().unwrap();
        assert!(widest <= 80, "QR is {widest} columns wide and will wrap");

        const QUIET: usize = 4;
        assert_eq!(widest, width + 2 * QUIET, "quiet zone is missing or the wrong size");

        let rows = modules_from_art(&art);
        assert!(rows.len() >= width + 2 * QUIET);

        // The quiet zone must be light all the way round, or a scanner cannot find
        // the symbol at all.
        for (top, above) in rows[0].iter().zip(&rows[QUIET - 1]) {
            assert!(!top && !above, "quiet zone is not light");
        }

        // Every module, in the right place and the right polarity. Inverting the
        // colours by mistake produces art that looks plausible and never scans.
        for y in 0..width {
            for x in 0..width {
                let expected = colors[y * width + x] == Color::Dark;
                assert_eq!(
                    rows[y + QUIET][x + QUIET],
                    expected,
                    "module ({x},{y}) does not match the encoder"
                );
            }
        }
    }

    /// A pairing payload can be long enough to push the symbol past a terminal.
    /// Version 12 is 65 modules plus the quiet zone, still inside 80 columns; past
    /// that the operator gets an unscannable smear with nothing saying why.
    #[test]
    fn test_a_long_payload_still_fits_the_terminal() {
        let payload = format!("2@{},{},{},1", "A".repeat(120), "B".repeat(90), "C".repeat(90));
        let code = QrCode::new(payload.as_bytes()).unwrap();
        let art = code
            .render::<unicode::Dense1x2>()
            .quiet_zone(true)
            .dark_color(unicode::Dense1x2::Light)
            .light_color(unicode::Dense1x2::Dark)
            .build();

        let widest = art.lines().map(|l| l.chars().count()).max().unwrap();
        assert!(widest <= 80, "a {}-byte payload renders {widest} columns wide", payload.len());
    }
}
