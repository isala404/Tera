use crate::codex::tier;
use crate::config::Config;
use crate::conversation::ConversationSession;
use crate::history::db::{ConversationEvent, HistoryDb, ProviderRef};
use crate::runtime::RuntimeDb;
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::{self as recurrence, ScheduleTiming};
use crate::secrets::SecretStore;
use crate::transport::{ReactionTarget, Transport};
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{error, info, warn};
use uuid::Uuid;

/// One-line JSON summary for logs; tool arguments can carry whole documents.
fn brief(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() <= 300 {
        return s;
    }
    let head: String = s.chars().take(300).collect();
    format!("{head}… (+{} more chars)", s.chars().count() - 300)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRpcRequest {
    pub id: u64,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRpcResponse {
    pub id: u64,
    pub result: Option<Value>,
    pub error: Option<String>,
}

pub struct DaemonRpcServer {
    config: Config,
    history_db: HistoryDb,
    runtime_db: RuntimeDb,
    transport: Arc<dyn Transport>,
    session: ConversationSession,
    secrets: SecretStore,
}

impl DaemonRpcServer {
    pub fn new(
        config: Config,
        history_db: HistoryDb,
        runtime_db: RuntimeDb,
        transport: Arc<dyn Transport>,
        session: ConversationSession,
    ) -> Self {
        Self {
            secrets: SecretStore::new(config.secrets_path()),
            config,
            history_db,
            runtime_db,
            transport,
            session,
        }
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let sock_path = self.config.socket_path();
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }

        if let Some(parent) = sock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = UnixListener::bind(&sock_path)
            .with_context(|| format!("Failed to bind Unix domain socket at {:?}", sock_path))?;
        info!("Daemon MCP RPC server listening on Unix socket {:?}", sock_path);

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let self_clone = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = self_clone.handle_client(stream).await {
                            error!("Error handling MCP socket client: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Unix socket accept error: {:?}", e);
                }
            }
        }
    }

    async fn handle_client(&self, stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        while buf_reader.read_line(&mut line).await? > 0 {
            if line.trim().is_empty() {
                line.clear();
                continue;
            }

            let req: DaemonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let err_resp = DaemonRpcResponse {
                        id: 0,
                        result: None,
                        error: Some(format!("Invalid RPC request payload: {:?}", e)),
                    };
                    let resp_str = serde_json::to_string(&err_resp)? + "\n";
                    writer.write_all(resp_str.as_bytes()).await?;
                    line.clear();
                    continue;
                }
            };

            // Both sides of every tool call are logged: this is the boundary
            // where the agent acts on the outside world.
            info!(
                target: "mcp::tool",
                "-> {}({})",
                req.tool_name,
                brief(&req.arguments)
            );
            let started = std::time::Instant::now();

            let response = match self.execute_tool(&req.tool_name, &req.arguments).await {
                Ok(val) => {
                    info!(
                        target: "mcp::tool",
                        "<- {} ok in {}ms: {}",
                        req.tool_name,
                        started.elapsed().as_millis(),
                        brief(&val)
                    );
                    DaemonRpcResponse {
                        id: req.id,
                        result: Some(val),
                        error: None,
                    }
                }
                Err(e) => {
                    warn!(
                        target: "mcp::tool",
                        "<- {} FAILED in {}ms: {e}",
                        req.tool_name,
                        started.elapsed().as_millis()
                    );
                    DaemonRpcResponse {
                        id: req.id,
                        result: None,
                        error: Some(e.to_string()),
                    }
                }
            };

            let resp_str = serde_json::to_string(&response)? + "\n";
            writer.write_all(resp_str.as_bytes()).await?;
            line.clear();
        }

        Ok(())
    }

    /// The chat to deliver into.
    ///
    /// Resolved from the live conversation rather than configuration: the
    /// configured owner value is a matching pattern for the gate, not a routable
    /// address, addressing it produced an unparseable JID, so every
    /// `send_message` failed and only the daemon's fallback text ever arrived.
    ///
    /// Scheduled tasks reuse the last known chat, which is what "message me the
    /// bitcoin price" means when the schedule fires later.
    fn recipient(&self) -> Result<String> {
        self.session.chat().ok_or_else(|| {
            anyhow!("No conversation to reply into yet, the assistant has not received a message")
        })
    }

    async fn execute_tool(&self, name: &str, args: &Value) -> Result<Value> {
        match name {
            "send_message" => {
                let recipient = self.recipient()?;
                let attachment = attachment_argument(args)?;

                // Two versions of the same message. `authored` is what the agent
                // wrote, with any credential it pasted rewritten out, and it is
                // what history keeps. `outgoing` has `${NAME}` filled in and goes
                // only to the wire, so an authorize URL reaches the owner working
                // while the value in it stays out of the transcript.
                //
                // Redact first, expand second. The other order would rewrite the
                // value straight back out of the message it just put there.
                let authored = args["text"].as_str().map(|raw| self.secrets.redact(raw));
                let outgoing = authored.as_deref().map(|text| self.secrets.expand(text));

                if authored.is_none() && attachment.is_none() {
                    return Err(anyhow!("send_message requires at least one of text, image_path, video_path, audio_path, or file_path"));
                }

                // Replying needs the provider's own id for the target, which is
                // what the provider ref table exists to translate back to.
                let reply_to = args["reply_to"].as_str();
                let reply_to_provider_id = match reply_to {
                    Some(event_id) => Some(
                        self.history_db
                            .lookup_provider_ref_by_event_id(event_id, "whatsapp")?
                            .ok_or_else(|| {
                                anyhow!("no WhatsApp message is recorded for event {event_id}; cannot reply to it")
                            })?
                            .provider_msg_id,
                    ),
                    None => None,
                };
                let reply_to_provider_id = reply_to_provider_id.as_deref();

                let provider_msg_id = if let Some((media_type, media_path_str)) = attachment {
                    let path = resolve_media_path(&self.config.workspace_dir, media_path_str)?;
                    self.transport
                        .send_media(
                            &recipient,
                            media_type,
                            &path,
                            outgoing.as_deref(),
                            reply_to_provider_id,
                        )
                        .await?
                } else {
                    self.transport
                        .send_text(
                            &recipient,
                            outgoing.as_deref().unwrap_or_default(),
                            reply_to_provider_id,
                        )
                        .await?
                };

                let event = ConversationEvent {
                    seq: None,
                    id: format!("m_{}", Uuid::new_v4().simple()),
                    occurred_at_ms: Utc::now().timestamp_millis(),
                    kind: "message".to_string(),
                    actor: "assistant".to_string(),
                    text: authored,
                    reply_to_id: reply_to.map(|id| id.to_string()),
                    turn_id: self.session.turn(),
                    reaction_target_id: None,
                    reaction_emoji: None,
                    attachments: vec![],
                };

                let saved_ev = self.history_db.insert_event(event)?;
                self.history_db
                    .record_provider_ref(&ProviderRef::whatsapp(
                        &saved_ev.id,
                        &provider_msg_id,
                        &recipient,
                        true,
                    ))?;
                self.history_db
                    .record_delivery_event(&saved_ev.id, "sent", None)?;

                // Tell the turn engine the agent already spoke, so it does not
                // deliver the final agent text on top of this.
                self.session.record_send();

                Ok(json!({
                    "status": "sent",
                    "message_id": saved_ev.id,
                    "provider_message_id": provider_msg_id
                }))
            }

            "react" => {
                let recipient = self.recipient()?;
                let msg_id = require_str(args, "message_id")?;
                let emoji = require_str(args, "emoji")?;

                // Reacting needs the chat and sender-side of the target, not just
                // its id; without them WhatsApp accepts the reaction and drops it.
                let stored = self
                    .history_db
                    .lookup_provider_ref_by_event_id(msg_id, "whatsapp")?
                    .ok_or_else(|| {
                        anyhow!("no WhatsApp message is recorded for event {msg_id}; cannot react to it")
                    })?;

                let target = ReactionTarget {
                    provider_msg_id: stored.provider_msg_id.clone(),
                    chat_jid: if stored.chat_jid.is_empty() {
                        recipient.clone()
                    } else {
                        stored.chat_jid.clone()
                    },
                    from_me: stored.from_me,
                };

                self.transport
                    .send_reaction(&recipient, &target, emoji)
                    .await?;

                let event = ConversationEvent {
                    seq: None,
                    id: format!("r_{}", Uuid::new_v4().simple()),
                    occurred_at_ms: Utc::now().timestamp_millis(),
                    kind: "reaction".to_string(),
                    actor: "assistant".to_string(),
                    text: None,
                    reply_to_id: None,
                    turn_id: None,
                    reaction_target_id: Some(msg_id.to_string()),
                    reaction_emoji: Some(emoji.to_string()),
                    attachments: vec![],
                };
                self.history_db.insert_event(event)?;

                Ok(json!({ "status": "reacted", "target": msg_id, "emoji": emoji }))
            }

            "schedule" => {
                let sched_name = require_str(args, "name")?;
                let prompt = require_str(args, "prompt")?;

                let timing = ScheduleTiming::parse(&args["timing"], Utc::now().timestamp_millis())?;

                // Most schedules are recurring checks, so the cheap tier is the
                // default and spending more is an explicit choice.
                let tier = match args["tier"].as_str() {
                    Some(name) => tier::by_name(name)?,
                    None => tier::ROUTINE,
                };

                let task_path = format!("tasks/schedule-{}", Uuid::new_v4().simple());

                let item = SchedulerDb::create_schedule(
                    &self.runtime_db,
                    sched_name,
                    prompt,
                    &timing,
                    &task_path,
                    tier,
                )?;

                // Echo the resolved time back, in the local time the rule was
                // written in, so the agent can see it means what it intended.
                Ok(json!({
                    "status": "scheduled",
                    "schedule_id": item.id,
                    "name": item.name,
                    "task_path": item.task_path,
                    "tier": item.tier,
                    "model": tier.model,
                    "first_run": recurrence::local_time(timing.first_run_ms),
                }))
            }

            "list_schedules" => {
                let items = SchedulerDb::list_schedules(&self.runtime_db)?;
                // Projected rather than dumped: the raw row carries epoch
                // milliseconds and the full task prompt, and the agent needs
                // neither to decide whether a schedule is the one it was looking
                // for. Local time, because that is what the rule means.
                let summaries: Vec<Value> = items
                    .iter()
                    .map(|item| {
                        json!({
                            "schedule_id": item.id,
                            "name": item.name,
                            "type": item.schedule_type,
                            "rrule": item.rrule,
                            "tier": item.tier,
                            "task_path": item.task_path,
                            "next_run": item.next_run_at_ms.map(recurrence::local_time),
                        })
                    })
                    .collect();
                Ok(json!(summaries))
            }

            "cancel_schedule" => {
                let id = require_str(args, "schedule_id")?;
                let success = SchedulerDb::cancel_schedule(&self.runtime_db, id)?;
                Ok(json!({ "cancelled": success, "schedule_id": id }))
            }

            "request_secret" => {
                let name = require_str(args, "name")?;
                let reason = require_str(args, "reason")?;
                let recipient = self.recipient()?;

                self.secrets
                    .request(name, reason, Utc::now().timestamp_millis())?;

                // The tool asks rather than returning instructions for the agent
                // to relay. The wording has to be exact, the owner's reply must be
                // the value and nothing else, and a paraphrase that invites "sure,
                // here you go: ..." stores the whole sentence as the credential.
                let ask = format!(
                    "I need {name} to {reason}. Send it as your next message, on its own with \
                     nothing else. I won't see it, it goes straight into the secret store. Then \
                     delete your message from this chat."
                );
                self.transport.send_text(&recipient, &ask, None).await?;
                self.session.record_send();

                Ok(json!({
                    "status": "requested",
                    "name": name,
                    "note": "The owner's next message becomes this value and you will not see it. \
                             You will get a note saying it arrived. The request expires in 15 minutes."
                }))
            }

            // Reading history is deliberately NOT a tool. The agent has a shell,
            // jq, sqlite3 and Python, and the projection is designed for them
            // (PLAN.md section 18). A tool here would only be a worse query
            // language than the ones it already knows.
            _ => Err(anyhow!("Unknown MCP tool name: '{}'", name)),
        }
    }
}

/// Reads a required string argument, erroring with a uniform message if it's
/// absent. Only for arguments that hard-error on absence; genuinely optional
/// ones stay as direct `.as_str()` reads.
fn require_str<'a>(args: &'a Value, field: &str) -> Result<&'a str> {
    args[field].as_str().ok_or_else(|| anyhow!("Missing {field}"))
}

fn attachment_argument(args: &Value) -> Result<Option<(&'static str, &str)>> {
    let candidates = [
        ("image", args["image_path"].as_str()),
        ("video", args["video_path"].as_str()),
        ("audio", args["audio_path"].as_str()),
        ("document", args["file_path"].as_str()),
    ];
    let supplied: Vec<_> = candidates
        .into_iter()
        .filter_map(|(media_type, path)| path.map(|path| (media_type, path)))
        .collect();

    if supplied.len() > 1 {
        return Err(anyhow!("send_message accepts only one attachment path per call"));
    }

    Ok(supplied.into_iter().next())
}

fn resolve_media_path(workspace: &Path, raw: &str) -> Result<PathBuf> {
    let raw = PathBuf::from(raw);
    let path = if raw.is_absolute() {
        raw
    } else {
        workspace.join(raw)
    };

    if !path.is_file() {
        return Err(anyhow!(
            "media file not found at {}; give a path to a file inside the workspace",
            path.display()
        ));
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn generic_file_is_sent_as_a_document() {
        let args = json!({"file_path": "notes.pdf"});
        let attachment = attachment_argument(&args).unwrap();
        assert_eq!(attachment, Some(("document", "notes.pdf")));
    }

    #[test]
    fn send_message_rejects_multiple_attachment_paths() {
        let error = attachment_argument(&json!({
            "image_path": "photo.jpg",
            "file_path": "notes.pdf"
        }))
        .unwrap_err();
        assert!(error.to_string().contains("only one attachment path"));
    }

    #[test]
    fn relative_and_absolute_files_resolve() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("notes.pdf");
        std::fs::write(&file, b"content").unwrap();

        assert_eq!(resolve_media_path(dir.path(), "notes.pdf").unwrap(), file);
        assert_eq!(resolve_media_path(dir.path(), file.to_str().unwrap()).unwrap(), file);
    }

    #[test]
    fn directories_are_not_sendable_files() {
        let dir = tempdir().unwrap();
        let error = resolve_media_path(dir.path(), ".").unwrap_err();
        assert!(error.to_string().contains("media file not found"));
    }

    /// A `DaemonRpcServer` wired to a mock transport, with a chat already open
    /// so `send_message` has somewhere to send.
    fn test_server(dir: &Path) -> (Arc<MockTransport>, DaemonRpcServer) {
        let config = Config::new(dir.to_path_buf(), true);
        for path in [config.history_db_path(), config.runtime_db_path(), config.secrets_path()] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        let history_db = HistoryDb::open_for(&config).unwrap();
        let runtime_db = RuntimeDb::open(&config.runtime_db_path()).unwrap();
        let transport = Arc::new(MockTransport::new());
        let session = ConversationSession::new();
        session.set_chat("owner@s.whatsapp.net");

        let server = DaemonRpcServer::new(config, history_db, runtime_db, transport.clone(), session);
        (transport, server)
    }

    /// The whole point of `${NAME}`: the owner gets a link that works, and the
    /// transcript the model reads back keeps only the placeholder.
    #[tokio::test]
    async fn a_secret_placeholder_is_filled_in_on_the_wire_but_not_in_history() {
        let dir = tempdir().unwrap();
        let (transport, server) = test_server(dir.path());
        server.secrets.set("SPOTIFY_CLIENT_ID", "abc123def456", 0).unwrap();

        server
            .execute_tool(
                "send_message",
                &json!({"text": "Log in: https://accounts.spotify.com/authorize?client_id=${SPOTIFY_CLIENT_ID}&state=x"}),
            )
            .await
            .unwrap();

        let sent = transport.sent_messages.lock().unwrap();
        assert!(sent[0].1.contains("client_id=abc123def456"), "{}", sent[0].1);

        let events = server.history_db.list_events_all().unwrap();
        let recorded = events[0].text.clone().unwrap();
        assert!(recorded.contains("client_id=${SPOTIFY_CLIENT_ID}"), "{recorded}");
        assert!(!recorded.contains("abc123def456"), "history kept the value: {recorded}");
    }

    /// A value the agent pasted itself still gets rewritten out on both paths.
    #[tokio::test]
    async fn a_pasted_secret_never_reaches_the_wire() {
        let dir = tempdir().unwrap();
        let (transport, server) = test_server(dir.path());
        server.secrets.set("SPOTIFY_CLIENT_ID", "abc123def456", 0).unwrap();

        server
            .execute_tool("send_message", &json!({"text": "your id is abc123def456"}))
            .await
            .unwrap();

        let sent = transport.sent_messages.lock().unwrap();
        assert_eq!(sent[0].1, "your id is [redacted SPOTIFY_CLIENT_ID]");
    }

    #[tokio::test]
    async fn reply_to_is_translated_into_the_provider_id() {
        let dir = tempdir().unwrap();
        let (transport, server) = test_server(dir.path());

        let incoming = server
            .history_db
            .insert_event(ConversationEvent {
                seq: None,
                id: String::new(),
                occurred_at_ms: 0,
                kind: "message".to_string(),
                actor: "user".to_string(),
                text: Some("which one?".to_string()),
                reply_to_id: None,
                turn_id: None,
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            })
            .unwrap();
        server
            .history_db
            .record_provider_ref(&ProviderRef::whatsapp(
                &incoming.id,
                "wamid.incoming",
                "owner@s.whatsapp.net",
                false,
            ))
            .unwrap();

        server
            .execute_tool("send_message", &json!({"text": "that one", "reply_to": incoming.id}))
            .await
            .unwrap();

        let sent = transport.sent_messages.lock().unwrap();
        assert_eq!(sent[0].2.as_deref(), Some("wamid.incoming"));

        let reply = server
            .history_db
            .list_events_all()
            .unwrap()
            .into_iter()
            .find(|e| e.actor == "assistant")
            .unwrap();
        assert_eq!(reply.reply_to_id, Some(incoming.id));
    }

    #[tokio::test]
    async fn replying_to_an_unknown_message_is_an_error() {
        let dir = tempdir().unwrap();
        let (_transport, server) = test_server(dir.path());

        let error = server
            .execute_tool("send_message", &json!({"text": "hi", "reply_to": "m_nope"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cannot reply to it"), "{error}");
    }
}
