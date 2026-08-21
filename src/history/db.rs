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
/// History is append-only and long-lived by design, so every column added from
/// here on needs a line in this function.
fn migrate(conn: &Connection) -> Result<()> {
    add_column_if_missing(conn, "provider_refs", "chat_jid", "TEXT")?;
    add_column_if_missing(conn, "provider_refs", "from_me", "INTEGER NOT NULL DEFAULT 0")?;
    Ok(())
}

fn add_column_if_missing(conn: &Connection, table: &str, column: &str, decl: &str) -> Result<()> {
    let existing: Vec<String> = conn
        .prepare(&format!("PRAGMA table_info({table})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<_, _>>()?;

    if existing.iter().any(|c| c == column) {
        return Ok(());
    }

    info!("Migrating {table}: adding column {column}");
    conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl};"))
        .with_context(|| format!("Failed to add {table}.{column}"))?;
    Ok(())
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEvent {
    pub seq: Option<i64>,
    pub id: String,
    pub occurred_at_ms: i64,
    pub kind: String,
    pub actor: String,
    pub text: Option<String>,
    pub reply_to_id: Option<String>,
    pub turn_id: Option<String>,
    pub reaction_target_id: Option<String>,
    pub reaction_emoji: Option<String>,
    pub attachments: Vec<Attachment>,
}

/// `get_event`, `list_events_all` and `recent_messages` each select the same
/// columns in the same order; sharing the mapper is what keeps them from
/// drifting when a column is added.
fn row_to_event(row: &Row) -> rusqlite::Result<ConversationEvent> {
    Ok(ConversationEvent {
        seq: Some(row.get(0)?),
        id: row.get(1)?,
        occurred_at_ms: row.get(2)?,
        kind: row.get(3)?,
        actor: row.get(4)?,
        text: row.get(5)?,
        reply_to_id: row.get(6)?,
        turn_id: row.get(7)?,
        reaction_target_id: row.get(8)?,
        reaction_emoji: row.get(9)?,
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
        migrate(&conn)?;
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
    /// A failed projection is not a failed write: SQLite has committed, so the
    /// event exists. The projection is marked dirty and rebuilt on next start.
    pub fn insert_event(&self, event: ConversationEvent) -> Result<ConversationEvent> {
        let event = self.insert_event_sqlite(event)?;

        if let Err(e) = ProjectionEngine::append_event(&self.jsonl_dir, &event) {
            error!(
                "Failed to append event {} to the JSONL projection: {:?}. \
                 Canonical history is intact; the projection will be rebuilt on next start.",
                event.id, e
            );
            ProjectionEngine::mark_dirty(&self.jsonl_dir);
        }

        Ok(event)
    }

    fn insert_event_sqlite(&self, mut event: ConversationEvent) -> Result<ConversationEvent> {
        let conn = self.conn.lock().unwrap();
        if event.id.is_empty() {
            event.id = format!("m_{}", Uuid::new_v4().simple());
        }
        if event.occurred_at_ms == 0 {
            event.occurred_at_ms = Utc::now().timestamp_millis();
        }

        conn.execute(
            "INSERT INTO conversation_events (
                id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                event.id,
                event.occurred_at_ms,
                event.kind,
                event.actor,
                event.text,
                event.reply_to_id,
                event.turn_id,
                event.reaction_target_id,
                event.reaction_emoji,
            ],
        )?;

        let seq = conn.last_insert_rowid();
        event.seq = Some(seq);

        for (pos, att) in event.attachments.iter_mut().enumerate() {
            att.event_id = event.id.clone();
            att.position = pos as i32;
            conn.execute(
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
            att.id = Some(conn.last_insert_rowid());
        }

        Ok(event)
    }

    /// Our event id for a message the provider knows by its own id.
    ///
    /// Reply targets arrive as WhatsApp ids. They have to be translated before
    /// they are stored, because history is addressed by event id and the JSONL
    /// projection must not contain provider ids at all (PLAN.md section 17.3) , 
    /// storing the raw provider id made `reply_to` unjoinable against anything.
    pub fn event_id_for_provider_ref(&self, provider: &str, provider_msg_id: &str) -> Result<Option<String>> {
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
            "INSERT OR REPLACE INTO provider_refs (event_id, provider, provider_message_id, chat_jid, from_me)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![r.event_id, r.provider, r.provider_msg_id, r.chat_jid, r.from_me as i32],
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

    pub fn record_delivery_event(&self, event_id: &str, state: &str, detail: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO delivery_events (event_id, occurred_at_ms, state, detail) VALUES (?1, ?2, ?3, ?4)",
            params![event_id, Utc::now().timestamp_millis(), state, detail],
        )?;
        Ok(())
    }

    pub fn get_event(&self, event_id: &str) -> Result<Option<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT seq, id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji
             FROM conversation_events WHERE id = ?1"
        )?;
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
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM conversation_events", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    pub fn list_events_all(&self) -> Result<Vec<ConversationEvent>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT seq, id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji
             FROM conversation_events ORDER BY seq ASC"
        )?;

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
        let mut stmt = conn.prepare(
            "SELECT seq, id, occurred_at_ms, kind, actor, text, reply_to_id, turn_id, reaction_target_id, reaction_emoji
             FROM conversation_events
             WHERE kind = 'message'
             ORDER BY seq DESC LIMIT ?1",
        )?;
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
}

#[cfg(test)]
mod migration_tests {
    use super::*;

    /// Regression: chat_jid and from_me were added to the schema, but
    /// `CREATE TABLE IF NOT EXISTS` left existing workspaces untouched, so every
    /// inbound message failed with "table provider_refs has no column chat_jid".
    #[test]
    fn test_opening_a_pre_migration_database_adds_missing_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.sqlite3");

        // A database as an older build would have left it.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE provider_refs (
                    event_id            TEXT NOT NULL,
                    provider            TEXT NOT NULL,
                    provider_message_id TEXT NOT NULL,
                    PRIMARY KEY(provider, provider_message_id)
                );",
            )
            .unwrap();
        }

        let db = HistoryDb::open(&path, &dir.path().join("jsonl")).unwrap();
        db.record_provider_ref(&ProviderRef::whatsapp(
            "evt_1",
            "wamid.1",
            "9477000@s.whatsapp.net",
            true,
        ))
        .unwrap();

        let stored = db
            .lookup_provider_ref_by_event_id("evt_1", "whatsapp")
            .unwrap()
            .expect("ref should round-trip");
        assert_eq!(stored.chat_jid, "9477000@s.whatsapp.net");
        assert!(stored.from_me);
    }

    #[test]
    fn test_migration_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.sqlite3");
        let jsonl = dir.path().join("jsonl");
        HistoryDb::open(&path, &jsonl).unwrap();
        HistoryDb::open(&path, &jsonl).unwrap();
    }

    /// Reply targets arrive as WhatsApp ids and have to be translated, or
    /// `reply_to` points at an id that appears nowhere in history and the JSONL
    /// leaks provider ids it is specified never to contain.
    #[test]
    fn test_provider_ids_translate_back_to_event_ids() {
        let dir = tempfile::tempdir().unwrap();
        let db = HistoryDb::open(
            &dir.path().join("history.sqlite3"),
            &dir.path().join("jsonl"),
        )
        .unwrap();

        db.record_provider_ref(&ProviderRef::whatsapp(
            "msg_first",
            "wamid.ABC",
            "947@s.whatsapp.net",
            false,
        ))
        .unwrap();

        assert_eq!(
            db.event_id_for_provider_ref("whatsapp", "wamid.ABC").unwrap(),
            Some("msg_first".to_string())
        );
        // A reply to something older than this workspace resolves to nothing,
        // which is the caller's cue to record no reply target at all.
        assert_eq!(
            db.event_id_for_provider_ref("whatsapp", "wamid.UNKNOWN").unwrap(),
            None
        );
    }

    /// Regression: `send_message` and `react` wrote events straight to SQLite and
    /// never appended to the projection, so the file the agent reads was missing
    /// every assistant message, 9 records against 23 canonical events on the
    /// live workspace.
    #[test]
    fn test_every_inserted_event_reaches_the_projection() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("jsonl");
        let db = HistoryDb::open(&dir.path().join("history.sqlite3"), &jsonl).unwrap();

        for (id, actor) in [("m_1", "user"), ("m_2", "assistant")] {
            db.insert_event(ConversationEvent {
                seq: None,
                id: id.to_string(),
                occurred_at_ms: 1_786_962_664_000,
                kind: "message".to_string(),
                actor: actor.to_string(),
                text: Some(format!("from {actor}")),
                reply_to_id: None,
                turn_id: None,
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            })
            .unwrap();
        }

        let projected = std::fs::read_to_string(jsonl.join("2026-08.jsonl")).unwrap();
        assert_eq!(projected.lines().count(), 2, "{projected}");
        assert!(projected.contains(r#""from":"assistant""#));
    }

    #[test]
    fn test_recent_messages_returns_ten_messages_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let db = HistoryDb::open(
            &dir.path().join("history.sqlite3"),
            &dir.path().join("jsonl"),
        )
        .unwrap();

        for index in 0..12 {
            db.insert_event(ConversationEvent {
                seq: None,
                id: format!("m_{index}"),
                occurred_at_ms: 1_786_962_664_000 + index,
                kind: "message".to_string(),
                actor: if index % 2 == 0 { "user" } else { "assistant" }.to_string(),
                text: Some(format!("message {index}")),
                reply_to_id: None,
                turn_id: None,
                reaction_target_id: None,
                reaction_emoji: None,
                attachments: vec![],
            })
            .unwrap();
        }

        let recent = db.recent_messages(10).unwrap();
        assert_eq!(recent.len(), 10);
        assert_eq!(recent.first().unwrap().id, "m_2");
        assert_eq!(recent.last().unwrap().id, "m_11");
        assert_eq!(recent[1].actor, "assistant");
    }
}
