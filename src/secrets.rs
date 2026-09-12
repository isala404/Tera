//! Credentials the owner sends through WhatsApp, held where the model never sees
//! them.
//!
//! Skills need API keys, and the only channel the owner has is a chat that feeds
//! straight into a model's context. Anything typed there is recorded in history,
//! projected to JSONL, replayed into later threads and folded into memory: four
//! places a key would sit in plain text forever. So the daemon
//! takes the value out of the message before any of that happens, stores it here,
//! and hands the model a note saying only that a secret arrived.
//!
//! The store lives under `.runtime/`, never inside a skill directory.
//! [`crate::workspace::init`] deletes and rewrites the skills tera ships on every
//! start, so a key written next to the skill that reads it would not survive an
//! update.
//!
//! What this is not: a sandbox. Codex runs with `danger-full-access`, so an agent
//! that decides to read `secrets.json` can. The property held here is narrower and
//! still worth having, the value is never *shown* to the model, so it cannot be
//! repeated back by accident, cannot be summarised into a memory, and does not
//! survive in a transcript. [`SecretStore::redact`] is the backstop for the day
//! something reads the file anyway.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// How long a request from [`SecretStore::request`] keeps claiming the next
/// message. Long enough for the owner to go and create a developer app, short
/// enough that a forgotten request does not swallow tomorrow's conversation.
pub const PENDING_LIFETIME_MS: i64 = 15 * 60 * 1000;

/// Shortest value [`SecretStore::redact`] will act on.
///
/// Redaction rewrites every occurrence of a stored value, so a four-character
/// secret would mangle ordinary prose containing those four characters. Real
/// credentials are long; anything this short is not worth corrupting messages
/// over.
const MIN_REDACTABLE_LEN: usize = 6;

/// Longest name accepted. Nothing depends on the bound, it just stops a typo
/// becoming a permanent key in the file.
const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Secret {
    pub value: String,
    pub set_at_ms: i64,
}

/// A credential the agent has asked for and not yet been given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRequest {
    pub name: String,
    pub requested_at_ms: i64,
}

impl PendingRequest {
    fn expired(&self, now_ms: i64) -> bool {
        now_ms - self.requested_at_ms > PENDING_LIFETIME_MS
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Contents {
    #[serde(default)]
    secrets: BTreeMap<String, Secret>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending: Option<PendingRequest>,
}

/// What an inbound message turned out to be.
///
/// Returned rather than acted on, so the parse can be tested without a store and
/// the caller keeps one place where a message stops being ordinary conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capture {
    /// Ordinary conversation. Pass it through untouched.
    Passthrough,
    /// A value now stored under this name.
    Stored { name: String },
    /// Meant as a secret, but unusable. The message still must not reach the
    /// model, so the caller reports this instead of the text.
    Rejected { reason: String },
}

/// Reads and writes `.runtime/secrets.json`.
///
/// Deliberately stateless: every call loads the file and every mutation writes it
/// back. The daemon and the `tera secret` CLI both hold one of these against the
/// same path, and a cached copy in either would let one overwrite the other's
/// work.
#[derive(Debug, Clone)]
pub struct SecretStore {
    path: PathBuf,
}

impl SecretStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load(&self) -> Result<Contents> {
        match fs::read_to_string(&self.path) {
            Ok(raw) => serde_json::from_str(&raw)
                .with_context(|| format!("{} is not valid JSON", self.path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Contents::default()),
            Err(error) => {
                Err(error).with_context(|| format!("Cannot read {}", self.path.display()))
            }
        }
    }

    /// 0600 from the start rather than chmod-ing after: a file that is briefly
    /// world-readable is world-readable. The directory is locked down too, so
    /// the store cannot be replaced out from under us by another user.
    fn save(&self, contents: &Contents) -> Result<()> {
        let parent = self
            .path
            .parent()
            .context("Secret store path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create {}", parent.display()))?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).ok();

        let mut serialized = serde_json::to_vec_pretty(contents)?;
        serialized.push(b'\n');
        crate::runtime::write_atomic(&self.path, &serialized, 0o600)
    }

    pub fn set(&self, name: &str, value: &str, now_ms: i64) -> Result<()> {
        let name = validate_name(name)?;
        if value.is_empty() {
            bail!("A secret cannot be empty");
        }
        let mut contents = self.load()?;
        contents.secrets.insert(
            name,
            Secret {
                value: value.to_string(),
                set_at_ms: now_ms,
            },
        );
        self.save(&contents)
    }

    pub fn get(&self, name: &str) -> Result<Option<String>> {
        Ok(self.load()?.secrets.remove(name).map(|secret| secret.value))
    }

    /// Every stored name, and when it was set. Never the values: this is what the
    /// agent is allowed to know, so it can tell whether it still needs to ask.
    pub fn names(&self) -> Result<Vec<(String, i64)>> {
        Ok(self
            .load()?
            .secrets
            .into_iter()
            .map(|(name, secret)| (name, secret.set_at_ms))
            .collect())
    }

    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut contents = self.load()?;
        let removed = contents.secrets.remove(name).is_some();
        if removed {
            self.save(&contents)?;
        }
        Ok(removed)
    }

    fn pending_path(&self) -> PathBuf {
        self.path.with_file_name("pending_secret.json")
    }

    /// Claim the owner's next message as the value for `name`.
    pub fn request(&self, name: &str, now_ms: i64) -> Result<()> {
        let name = validate_name(name)?;
        let mut contents = self.load()?;
        let req = PendingRequest {
            name: name.clone(),
            requested_at_ms: now_ms,
        };
        contents.pending = Some(req.clone());
        self.save(&contents)?;

        let parent = self
            .pending_path()
            .parent()
            .context("Secret store path has no parent directory")?
            .to_path_buf();
        fs::create_dir_all(&parent)?;
        let mut serialized = serde_json::to_vec_pretty(&req)?;
        serialized.push(b'\n');
        crate::runtime::write_atomic(&self.pending_path(), &serialized, 0o600)?;
        Ok(())
    }

    pub fn pending(&self, now_ms: i64) -> Result<Option<PendingRequest>> {
        let p_path = self.pending_path();
        if p_path.exists() {
            match fs::read_to_string(&p_path) {
                Ok(raw) => match serde_json::from_str::<PendingRequest>(&raw) {
                    Ok(req) => {
                        if !req.expired(now_ms) {
                            return Ok(Some(req));
                        }
                    }
                    Err(e) => {
                        bail!("Pending credential storage {:?} is corrupted: {e}", p_path);
                    }
                },
                Err(e) => {
                    bail!("Cannot read pending credential storage {:?}: {e}", p_path);
                }
            }
        }

        match self.load() {
            Ok(contents) => Ok(contents.pending.filter(|r| !r.expired(now_ms))),
            Err(e) => {
                if self.path.exists() {
                    bail!("Secret store {:?} is unreadable: {e}", self.path);
                }
                Ok(None)
            }
        }
    }

    pub fn has_pending(&self, now_ms: i64) -> bool {
        match self.pending(now_ms) {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(_) => true,
        }
    }

    pub fn clear_pending(&self) -> Result<()> {
        let _ = fs::remove_file(self.pending_path());
        if let Ok(mut contents) = self.load() {
            if contents.pending.is_some() {
                contents.pending = None;
                self.save(&contents)?;
            }
        }
        Ok(())
    }

    /// The one place a validation failure becomes a message for the owner, so
    /// the two ways in cannot report the same rejection differently.
    fn store(&self, name: String, value: &str, now_ms: i64) -> Capture {
        match self.set(&name, value, now_ms) {
            Ok(()) => Capture::Stored { name },
            Err(error) => Capture::Rejected {
                reason: error.to_string(),
            },
        }
    }

    /// Decide what an inbound message is, and store the value if it is a secret.
    ///
    /// Two ways in, one path out. `/secret NAME value` is unambiguous and always
    /// available. A pending request claims the whole next message, which is what
    /// makes the guided flow a matter of pasting a key rather than typing syntax
    /// correctly on a phone.
    pub fn capture(&self, text: &str, now_ms: i64) -> Result<Capture> {
        if let Some(rest) = command_argument(text) {
            let outcome = match parse_command(rest) {
                Ok((name, value)) => self.store(name, &value, now_ms),
                Err(reason) => Capture::Rejected { reason },
            };
            // Whatever happened, an explicit /secret ends any guided request:
            // leaving it armed would claim the owner's next ordinary message.
            self.clear_pending().ok();
            return Ok(outcome);
        }

        let pending = match self.pending(now_ms) {
            Ok(p) => p,
            Err(e) => {
                return Ok(Capture::Rejected {
                    reason: format!(
                        "a credential was requested, but the secret store could not be read: {e}"
                    ),
                });
            }
        };

        let Some(request) = pending else {
            return Ok(Capture::Passthrough);
        };

        let value = text.trim();
        let outcome = if value.is_empty() {
            Capture::Rejected {
                reason: format!("The message held no value for {}", request.name),
            }
        } else {
            self.store(request.name, value, now_ms)
        };
        self.clear_pending().ok();
        Ok(outcome)
    }

    /// Fill `${NAME}` references in `text` with the values they name.
    ///
    /// The mirror of [`SecretStore::redact`], and the reason that one can stay
    /// strict. Some credentials are meant to be sent: an OAuth authorize URL is
    /// useless to the owner without the client id in it, and redaction rewriting
    /// that id turned a link they were supposed to click into a broken one.
    ///
    /// So the agent writes the placeholder and never the value. Expansion happens
    /// on the way to the transport, after the message has been recorded, which
    /// keeps the value out of the model's context and out of history while the
    /// owner still receives something that works.
    ///
    /// Shell syntax rather than `{{NAME}}`, which the install-time template
    /// system already owns, and which reads right for names that are env-var
    /// shaped anyway. An unknown name is left alone: ordinary prose containing
    /// braces is not this feature's business to rewrite.
    pub fn expand(&self, text: &str) -> String {
        if !text.contains("${") {
            return text.to_string();
        }
        let Ok(contents) = self.load() else {
            return text.to_string();
        };

        let mut expanded = text.to_string();
        for (name, secret) in contents.secrets {
            let placeholder = format!("${{{name}}}");
            if expanded.contains(&placeholder) {
                expanded = expanded.replace(&placeholder, &secret.value);
            }
        }
        expanded
    }

    /// Replace every stored value in `text` with its name.
    ///
    /// The backstop, not the mechanism. Capture keeps secrets out of the model's
    /// context in the first place; this catches the case where an agent read the
    /// file itself and is about to put what it found into a message.
    ///
    /// Longest values first, so a secret that contains another one is not left
    /// half-rewritten.
    pub fn redact(&self, text: &str) -> String {
        let Ok(contents) = self.load() else {
            return text.to_string();
        };

        let mut values: Vec<(String, String)> = contents
            .secrets
            .into_iter()
            .filter(|(_, secret)| secret.value.len() >= MIN_REDACTABLE_LEN)
            .map(|(name, secret)| (secret.value, name))
            .collect();
        values.sort_by_key(|(value, _)| std::cmp::Reverse(value.len()));

        let mut redacted = text.to_string();
        for (value, name) in values {
            if redacted.contains(&value) {
                redacted = redacted.replace(&value, &format!("[redacted {name}]"));
            }
        }
        redacted
    }
}

/// Names are env-var shaped so a skill can use one directly as an environment
/// variable without a second naming convention to remember.
fn validate_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("A secret needs a name");
    }
    if name.len() > MAX_NAME_LEN {
        bail!("Secret names are at most {MAX_NAME_LEN} characters");
    }
    if !name.starts_with(|c: char| c.is_ascii_uppercase()) {
        bail!("Secret names start with an uppercase letter, like SPOTIFY_CLIENT_ID");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        bail!("Secret names use A-Z, 0-9 and underscore only, like SPOTIFY_CLIENT_ID");
    }
    Ok(name.to_string())
}

/// The text after `/secret`, if this message is that command.
///
/// Matched on the first line only. A pasted key sometimes arrives with trailing
/// lines of whatever the dashboard had next to it, and those belong to the value,
/// not to the command.
fn command_argument(text: &str) -> Option<&str> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("/secret")?;
    match rest.chars().next() {
        None => Some(""),
        Some(c) if c.is_whitespace() => Some(rest.trim_start()),
        // `/secretstuff` is a word, not this command.
        Some(_) => None,
    }
}

/// `NAME value` or `NAME=value`.
fn parse_command(rest: &str) -> Result<(String, String), String> {
    if rest.is_empty() {
        return Err("Usage: /secret NAME value".to_string());
    }

    let (name, value) = match rest.split_once(['=', ' ', '\t', '\n']) {
        Some((name, value)) => (name, value.trim()),
        None => (rest, ""),
    };

    let name = validate_name(name).map_err(|error| error.to_string())?;
    if value.is_empty() {
        return Err(format!(
            "No value came with {name}. Usage: /secret NAME value"
        ));
    }
    Ok((name, value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, SecretStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = SecretStore::new(dir.path().join("secrets.json"));
        (dir, store)
    }

    #[test]
    fn test_missing_file_is_an_empty_store() {
        let (_dir, store) = store();
        assert!(store.names().unwrap().is_empty());
        assert_eq!(store.get("NOPE").unwrap(), None);
        assert!(store.pending(0).unwrap().is_none());
    }

    #[test]
    fn test_set_and_get_round_trips() {
        let (_dir, store) = store();
        store.set("SPOTIFY_CLIENT_ID", "abc123", 10).unwrap();
        assert_eq!(
            store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(),
            Some("abc123")
        );
        assert_eq!(
            store.names().unwrap(),
            vec![("SPOTIFY_CLIENT_ID".to_string(), 10)]
        );
    }

    #[test]
    fn test_expand_fills_a_placeholder() {
        let (_dir, store) = store();
        store.set("SPOTIFY_CLIENT_ID", "abc123", 0).unwrap();
        assert_eq!(
            store.expand(
                "https://accounts.spotify.com/authorize?client_id=${SPOTIFY_CLIENT_ID}&x=1"
            ),
            "https://accounts.spotify.com/authorize?client_id=abc123&x=1"
        );
    }

    #[test]
    fn test_expand_leaves_unknown_names_alone() {
        let (_dir, store) = store();
        store.set("KNOWN", "value1", 0).unwrap();
        assert_eq!(
            store.expand("${UNKNOWN} and ${KNOWN}"),
            "${UNKNOWN} and value1"
        );
    }

    /// The pair has to compose in one direction only. Redaction runs first on
    /// what the agent wrote, expansion second on the way to the wire; doing it
    /// the other way round would rewrite the value straight back out again.
    #[test]
    fn test_redact_then_expand_leaves_a_usable_link() {
        let (_dir, store) = store();
        store.set("SPOTIFY_CLIENT_ID", "abc123def456", 0).unwrap();
        let authored = "Log in: https://spotify.com/auth?client_id=${SPOTIFY_CLIENT_ID}";
        let recorded = store.redact(authored);
        assert_eq!(recorded, authored, "a placeholder is not a value to redact");
        assert_eq!(
            store.expand(&recorded),
            "Log in: https://spotify.com/auth?client_id=abc123def456"
        );
    }

    /// The whole point of the file. A secret that lands world-readable has
    /// already leaked to every other account on the machine.
    #[test]
    fn test_store_is_private_to_the_owner() {
        let (_dir, store) = store();
        store.set("TOKEN", "value", 0).unwrap();
        let mode = fs::metadata(store.path()).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o077,
            0,
            "secrets.json must not be group or world readable"
        );
    }

    #[test]
    fn test_names_never_expose_values() {
        let (_dir, store) = store();
        store.set("TOKEN", "super-secret", 0).unwrap();
        let listed = format!("{:?}", store.names().unwrap());
        assert!(!listed.contains("super-secret"));
    }

    #[test]
    fn test_ordinary_message_passes_through() {
        let (_dir, store) = store();
        assert_eq!(
            store.capture("play some music", 0).unwrap(),
            Capture::Passthrough
        );
    }

    #[test]
    fn test_explicit_command_stores_the_value() {
        let (_dir, store) = store();
        let outcome = store
            .capture("/secret SPOTIFY_CLIENT_ID abc123", 0)
            .unwrap();
        assert_eq!(
            outcome,
            Capture::Stored {
                name: "SPOTIFY_CLIENT_ID".to_string()
            }
        );
        assert_eq!(
            store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn test_equals_form_stores_the_value() {
        let (_dir, store) = store();
        store
            .capture("/secret SPOTIFY_CLIENT_ID=abc123", 0)
            .unwrap();
        assert_eq!(
            store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(),
            Some("abc123")
        );
    }

    /// A word that merely starts with the command is conversation.
    #[test]
    fn test_similar_word_is_not_the_command() {
        let (_dir, store) = store();
        assert_eq!(
            store.capture("/secretly do the thing", 0).unwrap(),
            Capture::Passthrough
        );
    }

    /// A malformed command must still not fall through, or the value it was
    /// carrying reaches the model anyway.
    #[test]
    fn test_malformed_command_is_rejected_not_passed_through() {
        let (_dir, store) = store();
        assert!(matches!(
            store.capture("/secret SPOTIFY_CLIENT_ID", 0).unwrap(),
            Capture::Rejected { .. }
        ));
        assert!(matches!(
            store.capture("/secret lowercase value", 0).unwrap(),
            Capture::Rejected { .. }
        ));
    }

    #[test]
    fn test_pending_request_claims_the_next_message() {
        let (_dir, store) = store();
        store.request("SPOTIFY_CLIENT_ID", 0).unwrap();
        let outcome = store.capture("  abc123  ", 1_000).unwrap();
        assert_eq!(
            outcome,
            Capture::Stored {
                name: "SPOTIFY_CLIENT_ID".to_string()
            }
        );
        assert_eq!(
            store.get("SPOTIFY_CLIENT_ID").unwrap().as_deref(),
            Some("abc123")
        );
    }

    /// One message, one claim. Otherwise every later message is swallowed.
    #[test]
    fn test_request_is_consumed_by_the_message_it_claims() {
        let (_dir, store) = store();
        store.request("TOKEN", 0).unwrap();
        store.capture("abc123", 0).unwrap();
        assert_eq!(store.capture("thanks!", 0).unwrap(), Capture::Passthrough);
    }

    /// A request the owner never answered must not claim a message they send
    /// hours later about something else entirely.
    #[test]
    fn test_stale_request_does_not_claim_a_message() {
        let (_dir, store) = store();
        store.request("TOKEN", 0).unwrap();
        assert_eq!(
            store
                .capture("what's the weather", PENDING_LIFETIME_MS + 1)
                .unwrap(),
            Capture::Passthrough
        );
    }

    /// Answering a request with the explicit command must not leave the request
    /// armed to eat the following message.
    #[test]
    fn test_explicit_command_clears_a_pending_request() {
        let (_dir, store) = store();
        store.request("TOKEN", 0).unwrap();
        store.capture("/secret OTHER_TOKEN abc123", 0).unwrap();
        assert_eq!(store.capture("thanks!", 0).unwrap(), Capture::Passthrough);
    }

    #[test]
    fn test_redact_replaces_stored_values() {
        let (_dir, store) = store();
        store.set("TOKEN", "sk-live-abcdef", 0).unwrap();
        assert_eq!(
            store.redact("the key is sk-live-abcdef ok"),
            "the key is [redacted TOKEN] ok"
        );
    }

    /// A short value would rewrite ordinary prose that happens to contain it.
    #[test]
    fn test_redact_leaves_short_values_alone() {
        let (_dir, store) = store();
        store.set("PIN", "cat", 0).unwrap();
        assert_eq!(store.redact("the cat sat"), "the cat sat");
    }

    /// A secret that contains another must be rewritten whole, not left as a
    /// recognisable prefix around a redaction marker.
    #[test]
    fn test_redact_prefers_the_longer_value() {
        let (_dir, store) = store();
        store.set("SHORT", "abcdef", 0).unwrap();
        store.set("LONG", "abcdef-ghijkl", 0).unwrap();
        assert_eq!(
            store.redact("key abcdef-ghijkl here"),
            "key [redacted LONG] here"
        );
    }

    #[test]
    fn test_remove_reports_whether_anything_went() {
        let (_dir, store) = store();
        store.set("TOKEN", "value", 0).unwrap();
        assert!(store.remove("TOKEN").unwrap());
        assert!(!store.remove("TOKEN").unwrap());
    }

    #[test]
    fn test_pending_request_shared_across_distinct_instances() {
        let (dir, store1) = store();
        let store2 = SecretStore::new(dir.path().join("secrets.json"));

        store1.request("API_KEY", 0).unwrap();
        assert!(store2.has_pending(0));

        let outcome = store2.capture("my-secret-key-12345", 100).unwrap();
        assert_eq!(
            outcome,
            Capture::Stored {
                name: "API_KEY".to_string(),
            }
        );
        assert!(!store1.has_pending(100));
        assert!(!store2.has_pending(100));
    }

    #[test]
    fn test_corrupt_store_fails_closed_and_does_not_pass_through() {
        let (dir, store) = store();
        let path = dir.path().join("secrets.json");
        std::fs::write(&path, "CORRUPT NOT JSON").unwrap();

        // Corrupt store must fail closed: has_pending returns true, capture returns Rejected
        assert!(store.has_pending(0));
        let capture = store.capture("potential-secret-leak", 0).unwrap();
        assert!(matches!(capture, Capture::Rejected { .. }));
    }

    #[test]
    fn test_request_on_corrupt_store_fails_without_erasing_store() {
        let (dir, store) = store();
        let path = dir.path().join("secrets.json");
        let corrupt_content = "CORRUPT NOT JSON CONTENT";
        std::fs::write(&path, corrupt_content).unwrap();

        let res = store.request("NEW_KEY", 0);
        assert!(res.is_err(), "request on corrupt store must return Err");
        let content_after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content_after, corrupt_content,
            "store was erased or overwritten"
        );
    }
}
