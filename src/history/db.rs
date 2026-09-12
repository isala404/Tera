use crate::config::Config;
use crate::history::projection::ProjectionEngine;
use crate::history::schema::INIT_HISTORY_SCHEMA_SQL;
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use uuid::Uuid;

/// Bring an existing database up to the current schema.
///
/// `CREATE TABLE IF NOT EXISTS` is a no-op on a table that already exists, so
/// adding a column to the schema does nothing to workspaces created before it.
/// Link between an internal event and the provider's id for the same message.
///
/// A named struct rather than three `&str` parameters: the positional form was
/// called with the arguments in two different orders, which silently wrote
/// `event_id = "whatsapp"` and made the mapping unusable in both directions.
#[derive(Debug, Clone)]
pub struct ProviderRef {
    pub event_id: String,
    pub provider: String,
    pub provider_msg_id: String,
    /// Chat the message lives in, without a device suffix.
    pub chat_jid: String,
    pub from_me: bool,
}

impl ProviderRef {
    /// The only provider today; keeps call sites from spelling it by hand.
    pub fn whatsapp(
        event_id: impl Into<String>,
        provider_msg_id: impl Into<String>,
        chat_jid: impl Into<String>,
        from_me: bool,
    ) -> Self {
        Self {
            event_id: event_id.into(),
            provider: "whatsapp".to_string(),
            provider_msg_id: provider_msg_id.into(),
            chat_jid: chat_jid.into(),
            from_me,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub id: Option<i64>,
    pub event_id: String,
    pub position: i32,
    pub media_type: String,
    pub relative_path: String,
    pub mime_type: Option<String>,
    pub original_name: Option<String>,
}

/// What a row in `events` is.
///
/// Stored as the lowercase name because the agent queries the `kind` column
/// with `sqlite3` directly, so the text is part of the interface, not an
/// implementation detail we are free to rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    Message,
    Reaction,
}

impl EventKind {
    fn as_str(self) -> &'static str {
        match self {
            EventKind::Message => "message",
            EventKind::Reaction => "reaction",
        }
    }
}

impl rusqlite::ToSql for EventKind {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl rusqlite::types::FromSql for EventKind {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        match value.as_str()? {
            "message" => Ok(EventKind::Message),
            "reaction" => Ok(EventKind::Reaction),
            other => Err(rusqlite::types::FromSqlError::Other(
                format!("unknown event kind {other:?}").into(),
            )),
        }
    }
}

pub(crate) const EVENT_COLUMNS: &str =
    "seq, id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventPayload {
    Message {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reply_to_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    Reaction {
        target_id: String,
        emoji: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEvent {
    pub seq: Option<i64>,
    pub id: String,
    pub occurred_at_ms: i64,
    pub actor: String,
    pub payload: EventPayload,
    pub attachments: Vec<Attachment>,
}

impl ConversationEvent {
    pub fn message(
        id: impl Into<String>,
        occurred_at_ms: i64,
        actor: impl Into<String>,
        text: Option<String>,
        reply_to_id: Option<String>,
        turn_id: Option<String>,
        attachments: Vec<Attachment>,
    ) -> Self {
        Self {
            seq: None,
            id: id.into(),
            occurred_at_ms,
            actor: actor.into(),
            payload: EventPayload::Message {
                text,
                reply_to_id,
                turn_id,
            },
            attachments,
        }
    }

    pub fn reaction(
        id: impl Into<String>,
        occurred_at_ms: i64,
        actor: impl Into<String>,
        target_id: impl Into<String>,
        emoji: impl Into<String>,
    ) -> Self {
        Self {
            seq: None,
            id: id.into(),
            occurred_at_ms,
            actor: actor.into(),
            payload: EventPayload::Reaction {
                target_id: target_id.into(),
                emoji: emoji.into(),
            },
            attachments: vec![],
        }
    }

    pub fn kind(&self) -> EventKind {
        match &self.payload {
            EventPayload::Message { .. } => EventKind::Message,
            EventPayload::Reaction { .. } => EventKind::Reaction,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match &self.payload {
            EventPayload::Message { text, .. } => text.as_deref(),
            EventPayload::Reaction { .. } => None,
        }
    }

    pub fn reply_to_id(&self) -> Option<&str> {
        match &self.payload {
            EventPayload::Message { reply_to_id, .. } => reply_to_id.as_deref(),
            EventPayload::Reaction { .. } => None,
        }
    }

    pub fn turn_id(&self) -> Option<&str> {
        match &self.payload {
            EventPayload::Message { turn_id, .. } => turn_id.as_deref(),
            EventPayload::Reaction { .. } => None,
        }
    }

    pub fn reaction_target_id(&self) -> Option<&str> {
        match &self.payload {
            EventPayload::Message { .. } => None,
            EventPayload::Reaction { target_id, .. } => Some(target_id.as_str()),
        }
    }

    pub fn reaction_emoji(&self) -> Option<&str> {
        match &self.payload {
            EventPayload::Message { .. } => None,
            EventPayload::Reaction { emoji, .. } => Some(emoji.as_str()),
        }
    }
}

/// `get_event`, `list_events_all`, `recent_messages` and `messages_for_turn` each
/// select the same columns in the same order; sharing the mapper is what keeps
/// them from drifting when a column is added.
fn row_to_event(row: &Row) -> rusqlite::Result<ConversationEvent> {
    let seq: Option<i64> = Some(row.get(0)?);
    let id: String = row.get(1)?;
    let occurred_at_ms: i64 = row.get(2)?;
    let kind: EventKind = row.get(3)?;
    let actor: String = row.get(4)?;
    let text: Option<String> = row.get(5)?;
    let reply_to_id: Option<String> = row.get(6)?;
    let turn_id: Option<String> = row.get(7)?;
    let reaction_target_id: Option<String> = row.get(8)?;
    let reaction_emoji: Option<String> = row.get(9)?;

    let payload = match kind {
        EventKind::Message => EventPayload::Message {
            text,
            reply_to_id,
            turn_id,
        },
        EventKind::Reaction => {
            let target_id = reaction_target_id
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        8,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "malformed reaction event: missing reaction_target_id",
                        )),
                    )
                })?;
            let emoji = reaction_emoji.filter(|s| !s.is_empty()).ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    9,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "malformed reaction event: missing reaction_emoji",
                    )),
                )
            })?;
            EventPayload::Reaction { target_id, emoji }
        }
    };

    Ok(ConversationEvent {
        seq,
        id,
        occurred_at_ms,
        actor,
        payload,
        attachments: vec![],
    })
}

fn load_attachments(conn: &Connection, event_id: &str) -> Result<Vec<Attachment>> {
    let mut stmt = conn.prepare(
        "SELECT id, event_id, position, media_type, relative_path, mime_type, original_name
         FROM attachments WHERE event_id = ?1 ORDER BY position ASC",
    )?;
    let rows = stmt.query_map(params![event_id], |row| {
        Ok(Attachment {
            id: Some(row.get(0)?),
            event_id: row.get(1)?,
            position: row.get(2)?,
            media_type: row.get(3)?,
            relative_path: row.get(4)?,
            mime_type: row.get(5)?,
            original_name: row.get(6)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// The canonical event store, and the owner of its JSONL projection.
///
/// The projection is written here rather than by callers. When appending was the
/// caller's job, the two MCP tool paths forgot to do it and every assistant
/// message and reaction was missing from the file the agent actually reads.
#[derive(Clone)]
pub struct HistoryDb {
    conn: Arc<Mutex<Connection>>,
    jsonl_dir: PathBuf,
}

impl HistoryDb {
    pub fn open(db_path: &Path, jsonl_dir: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open SQLite history DB at {:?}", db_path))?;
        conn.execute_batch(INIT_HISTORY_SCHEMA_SQL)?;
        info!("Opened history database at {:?}", db_path);
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            jsonl_dir: jsonl_dir.to_path_buf(),
        })
    }

    /// The database and its projection are one unit; this keeps every call site
    /// from having to remember both halves.
    pub fn open_for(config: &Config) -> Result<Self> {
        Self::open(&config.history_db_path(), &config.history_jsonl_dir())
    }

    /// Append the event to canonical history, then project it into JSONL.
    ///
    /// Wrapped in a SQLite transaction together with any dependent rows: if an
    /// attachment or provider reference fails to write, no orphaned event row or
    /// projection survives.
    pub fn insert_event(&self, event: ConversationEvent) -> Result<ConversationEvent> {
        self.insert_event_full(event, None, None)
            .map(|opt| opt.expect("event without provider ref cannot be deduplicated away"))
    }

    /// Insert an inbound event and its provider reference atomically.
    ///
    /// Returns `Ok(None)` if the provider reference has already been accepted,
    /// deduplicating replayed messages before side effects can happen.
    pub fn insert_inbound_event(
        &self,
        event: ConversationEvent,
        provider_ref: ProviderRef,
    ) -> Result<Option<ConversationEvent>> {
        self.insert_event_full(event, Some(&provider_ref), None)
    }

    pub fn insert_event_full(
        &self,
        mut event: ConversationEvent,
        provider_ref: Option<&ProviderRef>,
        delivery: Option<(&str, Option<&str>)>,
    ) -> Result<Option<ConversationEvent>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        if let Some(r) = provider_ref {
            let mut check_stmt = tx.prepare(
                "SELECT 1 FROM provider_refs WHERE provider = ?1 AND provider_message_id = ?2",
            )?;
            let exists = check_stmt
                .query_row(params![r.provider, r.provider_msg_id], |_| Ok(()))
                .optional()?
                .is_some();
            if exists {
                return Ok(None);
            }
        }

        if event.id.is_empty() {
            event.id = format!("m_{}", Uuid::new_v4().simple());
        }
        if event.occurred_at_ms == 0 {
            event.occurred_at_ms = Utc::now().timestamp_millis();
        }

        let (kind, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji) =
            match &event.payload {
                EventPayload::Message {
                    text,
                    reply_to_id,
                    turn_id,
                } => (
                    EventKind::Message,
                    text.as_deref(),
                    reply_to_id.as_deref(),
                    turn_id.as_deref(),
                    None,
                    None,
                ),
                EventPayload::Reaction { target_id, emoji } => (
                    EventKind::Reaction,
                    None,
                    None,
                    None,
                    Some(target_id.as_str()),
                    Some(emoji.as_str()),
                ),
            };

        tx.execute(
            "INSERT INTO conversation_events (
                id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                event.id,
                event.occurred_at_ms,
                kind,
                event.actor,
                text,
                reply_to_id,
                turn_id,
                reaction_target_id,
                reaction_emoji,
            ],
        )?;

        let seq = tx.last_insert_rowid();
        event.seq = Some(seq);

        for (pos, att) in event.attachments.iter_mut().enumerate() {
            att.event_id = event.id.clone();
            att.position = pos as i32;
            tx.execute(
                "INSERT INTO attachments (
                    event_id, position, media_type, relative_path, mime_type, original_name
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    att.event_id,
                    att.position,
                    att.media_type,
                    att.relative_path,
                    att.mime_type,
                    att.original_name,
                ],
            )?;
            att.id = Some(tx.last_insert_rowid());
        }

        if let Some(r) = provider_ref {
            tx.execute(
                "INSERT INTO provider_refs (event_id, provider, provider_message_id, chat_jid, from_me)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![event.id, r.provider, r.provider_msg_id, r.chat_jid, r.from_me as i32],
            )?;
        }

        if let Some((state, detail)) = delivery {
            tx.execute(
                "INSERT INTO delivery_events (event_id, occurred_at_ms, state, detail) VALUES (?1, ?2, ?3, ?4)",
                params![event.id, Utc::now().timestamp_millis(), state, detail],
            )?;
        }

        tx.commit()?;
        drop(conn);

        if let Err(e) = ProjectionEngine::append_event(&self.jsonl_dir, &event) {
            error!(
                "Failed to append event {} to the JSONL projection: {:?}. \
                 Canonical history is intact; the projection will be rebuilt on next start.",
                event.id, e
            );
            ProjectionEngine::mark_dirty(&self.jsonl_dir);
        }

        Ok(Some(event))
    }

    /// Whether this provider message has already been recorded in history.
    pub fn is_provider_message_recorded(
        &self,
        provider: &str,
        provider_msg_id: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT 1 FROM provider_refs WHERE provider = ?1 AND provider_message_id = ?2",
        )?;
        let exists = stmt
            .query_row(params![provider, provider_msg_id], |_| Ok(()))
            .optional()?
            .is_some();
        Ok(exists)
    }

    /// Our event id for a message the provider knows by its own id.
    ///
    /// Reply targets arrive as WhatsApp ids. They have to be translated before
    /// they are stored, because history is addressed by event id and the JSONL
    /// projection must not contain provider ids at all ,
    /// storing the raw provider id made `reply_to` unjoinable against anything.
    pub fn event_id_for_provider_ref(
        &self,
        provider: &str,
        provider_msg_id: &str,
    ) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT event_id FROM provider_refs WHERE provider = ?1 AND provider_message_id = ?2",
        )?;
        let res = stmt
            .query_row(params![provider, provider_msg_id], |row| row.get(0))
            .optional()?;
        Ok(res)
    }

    pub fn record_provider_ref(&self, r: &ProviderRef) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO provider_refs (event_id, provider, provider_message_id, chat_jid, from_me)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                r.event_id,
                r.provider,
                r.provider_msg_id,
                r.chat_jid,
                r.from_me as i32
            ],
        )?;
        Ok(())
    }

    pub fn lookup_provider_ref_by_event_id(
        &self,
        event_id: &str,
        provider: &str,
    ) -> Result<Option<ProviderRef>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT provider_message_id, chat_jid, from_me FROM provider_refs
             WHERE event_id = ?1 AND provider = ?2",
        )?;
        let res = stmt
            .query_row(params![event_id, provider], |row| {
                let from_me: i32 = row.get(2)?;
                Ok(ProviderRef {
                    event_id: event_id.to_string(),
                    provider: provider.to_string(),
                    provider_msg_id: row.get(0)?,
                    chat_jid: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    from_me: from_me != 0,
                })
            })
            .optional()?;
        Ok(res)
    }

    pub fn lookup_provider_ref_by_provider_id(
        &self,
        provider_msg_id: &str,
        provider: &str,
    ) -> Result<Option<ProviderRef>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT event_id, chat_jid, from_me FROM provider_refs
             WHERE provider_message_id = ?1 AND provider = ?2",
        )?;
        let res = stmt
            .query_row(params![provider_msg_id, provider], |row| {
                let from_me: i32 = row.get(2)?;
                Ok(ProviderRef {
                    event_id: row.get(0)?,
                    provider: provider.to_string(),
                    provider_msg_id: provider_msg_id.to_string(),
                    chat_jid: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    from_me: from_me != 0,
                })
            })
            .optional()?;
        Ok(res)
    }

    pub fn list_turn_events(&self, turn_id: &str) -> Result<Vec<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let query = format!(
            "SELECT {EVENT_COLUMNS} FROM conversation_events WHERE turn_id = ?1 ORDER BY seq ASC"
        );
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params![turn_id], row_to_event)?;
        let mut events = Vec::new();
        for row in rows {
            let mut ev = row?;
            ev.attachments = load_attachments(&conn, &ev.id)?;
            events.push(ev);
        }
        Ok(events)
    }

    pub fn record_delivery_event(
        &self,
        event_id: &str,
        state: &str,
        detail: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO delivery_events (event_id, occurred_at_ms, state, detail) VALUES (?1, ?2, ?3, ?4)",
            params![event_id, Utc::now().timestamp_millis(), state, detail],
        )?;
        Ok(())
    }

    pub fn get_event(&self, event_id: &str) -> Result<Option<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let query = format!("SELECT {EVENT_COLUMNS} FROM conversation_events WHERE id = ?1");
        let mut stmt = conn.prepare(&query)?;
        let event_opt = stmt.query_row(params![event_id], row_to_event).optional()?;

        if let Some(mut ev) = event_opt {
            ev.attachments = load_attachments(&conn, event_id)?;
            Ok(Some(ev))
        } else {
            Ok(None)
        }
    }

    pub fn count_events(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM conversation_events", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    pub fn list_events_all(&self) -> Result<Vec<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let query = format!("SELECT {EVENT_COLUMNS} FROM conversation_events ORDER BY seq ASC");
        let mut stmt = conn.prepare(&query)?;

        let mut events = Vec::new();
        let rows = stmt.query_map([], row_to_event)?;

        for ev in rows {
            let mut ev = ev?;
            ev.attachments = load_attachments(&conn, &ev.id)?;
            events.push(ev);
        }

        Ok(events)
    }

    /// Return the most recent messages in conversation order. This is the small
    /// recovery window used when a Codex thread has expired, so a new thread can
    /// rejoin the conversation without receiving the whole history database.
    pub fn recent_messages(&self, limit: usize) -> Result<Vec<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let query = format!(
            "SELECT {EVENT_COLUMNS} FROM conversation_events
             WHERE kind = 'message'
             ORDER BY seq DESC LIMIT ?1"
        );
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params![limit as i64], row_to_event)?;

        let mut events = Vec::new();
        for row in rows {
            let mut event = row?;
            event.attachments = load_attachments(&conn, &event.id)?;
            events.push(event);
        }

        events.reverse();
        Ok(events)
    }

    /// Return exactly the messages belonging to one logical conversation turn.
    /// Recovery must not mistake nearby history for part of the interrupted job.
    pub fn messages_for_turn(&self, turn_id: &str) -> Result<Vec<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let query = format!(
            "SELECT {EVENT_COLUMNS} FROM conversation_events
             WHERE kind = 'message' AND turn_id = ?1
             ORDER BY seq ASC"
        );
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(params![turn_id], row_to_event)?;

        let mut events = Vec::new();
        for row in rows {
            let mut event = row?;
            event.attachments = load_attachments(&conn, &event.id)?;
            events.push(event);
        }
        Ok(events)
    }
}
