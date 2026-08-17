//! Who is allowed to drive the assistant.
//!
//! The daemon runs Codex with `approval_policy = never` and full disk access on
//! the owner's machine. Anyone whose message reaches the turn engine can
//! therefore run arbitrary work as the owner, so the check belongs in one place
//! with one answer, not spread across the transport as ad-hoc `if`s.
//!
//! PLAN.md section 81: single owner in V1, reject unknown senders.

use crate::transport::InboundMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Accept,
    Reject(RejectReason),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    NotOwner,
    GroupChat,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::NotOwner => write!(f, "sender is not the configured owner"),
            RejectReason::GroupChat => write!(f, "group chats are not served"),
        }
    }
}

/// Decides whether an inbound message may start a turn.
#[derive(Debug, Clone)]
pub struct OwnerPolicy {
    /// Owner JID as configured, e.g. `94771234567@s.whatsapp.net`, a bare
    /// number, or a `@lid` identifier. Compared on the user part only, because
    /// WhatsApp varies the server and appends a device suffix.
    owner_jid: Option<String>,
}

impl OwnerPolicy {
    pub fn new(owner_jid: Option<String>) -> Self {
        Self { owner_jid }
    }

    pub fn evaluate(&self, msg: &InboundMessage) -> Verdict {
        if msg.is_group {
            return Verdict::Reject(RejectReason::GroupChat);
        }

        // With no explicit owner configured, the owner is the account this daemon
        // is paired to, messaging from any of its own devices. That keeps the
        // default closed rather than open, and matches the normal setup: pair a
        // linked device, then talk to it from your phone.
        let Some(owner) = &self.owner_jid else {
            return if msg.from_own_account {
                Verdict::Accept
            } else {
                Verdict::Reject(RejectReason::NotOwner)
            };
        };

        if msg.from_own_account || jid_user(&msg.sender) == jid_user(owner) {
            Verdict::Accept
        } else {
            Verdict::Reject(RejectReason::NotOwner)
        }
    }
}

/// The user part of a JID: `94771234567:12@s.whatsapp.net` -> `94771234567`.
///
/// WhatsApp appends a device suffix and varies the server between `@lid` and
/// `@s.whatsapp.net` for the same person, so only the user part is comparable.
pub fn jid_user(jid: &str) -> &str {
    let no_server = jid.split('@').next().unwrap_or(jid);
    no_server.split(':').next().unwrap_or(no_server)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(sender: &str, from_own_account: bool, is_group: bool) -> InboundMessage {
        InboundMessage {
            provider_msg_id: "m1".into(),
            sender: sender.into(),
            text: Some("hi".into()),
            timestamp_ms: 0,
            reply_to_provider_msg_id: None,
            media_attachment: None,
            chat_jid: sender.split(':').next().unwrap_or(sender).to_string(),
            from_own_account,
            is_group,
        }
    }

    #[test]
    fn test_unconfigured_policy_accepts_only_the_paired_account() {
        let policy = OwnerPolicy::new(None);
        assert_eq!(policy.evaluate(&msg("254910671147212@lid", true, false)), Verdict::Accept);
        assert_eq!(
            policy.evaluate(&msg("94770000000@s.whatsapp.net", false, false)),
            Verdict::Reject(RejectReason::NotOwner)
        );
    }

    #[test]
    fn test_configured_owner_matches_ignoring_device_and_server() {
        let policy = OwnerPolicy::new(Some("94771234567".into()));
        assert_eq!(
            policy.evaluate(&msg("94771234567:12@s.whatsapp.net", false, false)),
            Verdict::Accept
        );
        assert_eq!(
            policy.evaluate(&msg("94779999999@s.whatsapp.net", false, false)),
            Verdict::Reject(RejectReason::NotOwner)
        );
    }

    /// Regression: a message from the owner's own phone arrives with a device
    /// suffix (`...:26@lid`) and is_from_me = false, because this device did not
    /// send it. Matching on the paired account is what makes it work.
    #[test]
    fn test_owner_phone_with_device_suffix_is_accepted() {
        let policy = OwnerPolicy::new(None);
        assert_eq!(
            policy.evaluate(&msg("254910671147212:26@lid", true, false)),
            Verdict::Accept
        );

        let configured = OwnerPolicy::new(Some("254910671147212@lid".into()));
        assert_eq!(
            configured.evaluate(&msg("254910671147212:26@lid", false, false)),
            Verdict::Accept
        );
    }

    #[test]
    fn test_group_chats_are_never_served() {
        let policy = OwnerPolicy::new(Some("94771234567".into()));
        assert_eq!(
            policy.evaluate(&msg("94771234567@s.whatsapp.net", true, true)),
            Verdict::Reject(RejectReason::GroupChat)
        );
    }
}
