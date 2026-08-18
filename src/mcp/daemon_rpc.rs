use crate::codex::tier;
use crate::config::Config;
use crate::conversation::ConversationSession;
use crate::history::db::{ConversationEvent, HistoryDb, ProviderRef};
use crate::runtime::RuntimeDb;
use crate::scheduler::db::SchedulerDb;
use crate::scheduler::recurrence::{self as recurrence, ScheduleTiming};
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
                let text = args["text"].as_str();
                let attachment = attachment_argument(args)?;

                if text.is_none() && attachment.is_none() {
                    return Err(anyhow!("send_message requires at least one of text, image_path, video_path, audio_path, or file_path"));
                }

                let provider_msg_id = if let Some((media_type, media_path_str)) = attachment {
                    let path = resolve_media_path(&self.config.workspace_dir, media_path_str)?;
                    self.transport
                        .send_media(&recipient, media_type, &path, text, None)
                        .await?
                } else {
                    let text_str = text.unwrap_or_default();
                    self.transport.send_text(&recipient, text_str, None).await?
                };

                // Save assistant event to history
                let event = ConversationEvent {
                    seq: None,
                    id: format!("m_{}", Uuid::new_v4().simple()),
                    occurred_at_ms: Utc::now().timestamp_millis(),
                    kind: "message".to_string(),
                    actor: "assistant".to_string(),
                    text: text.map(|s| s.to_string()),
                    reply_to_id: None,
                    turn_id: None,
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
                let msg_id = args["message_id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing message_id parameter"))?;
                let emoji = args["emoji"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing emoji parameter"))?;

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
                let sched_name = args["name"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing schedule name"))?;
                let prompt = args["prompt"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing schedule prompt"))?;

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
                let id = args["schedule_id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing schedule_id"))?;
                let success = SchedulerDb::cancel_schedule(&self.runtime_db, id)?;
                Ok(json!({ "cancelled": success, "schedule_id": id }))
            }

            // Reading history is deliberately NOT a tool. The agent has a shell,
            // jq, sqlite3 and Python, and the projection is designed for them
            // (PLAN.md section 18). A tool here would only be a worse query
            // language than the ones it already knows.
            _ => Err(anyhow!("Unknown MCP tool name: '{}'", name)),
        }
    }
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
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn generic_file_is_sent_as_a_document() {
        let attachment = attachment_argument(&json!({"file_path": "notes.pdf"})).unwrap();
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
}
