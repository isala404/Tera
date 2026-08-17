pub const INIT_HISTORY_SCHEMA_SQL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversation_events (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    id                 TEXT NOT NULL UNIQUE,
    occurred_at_ms     INTEGER NOT NULL,
    kind               TEXT NOT NULL,
    actor              TEXT NOT NULL,
    text               TEXT,
    reply_to_id        TEXT,
    turn_id            TEXT,
    reaction_target_id TEXT,
    reaction_emoji     TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_time
ON conversation_events(occurred_at_ms);

CREATE INDEX IF NOT EXISTS idx_events_turn
ON conversation_events(turn_id);

CREATE INDEX IF NOT EXISTS idx_events_reply
ON conversation_events(reply_to_id);

CREATE TABLE IF NOT EXISTS attachments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    position       INTEGER NOT NULL,
    media_type     TEXT NOT NULL,
    relative_path  TEXT NOT NULL,
    mime_type      TEXT,
    original_name  TEXT,
    FOREIGN KEY(event_id) REFERENCES conversation_events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attachments_event
ON attachments(event_id);

-- chat_jid and from_me are stored alongside the id because WhatsApp keys a
-- reaction by all three; an id on its own cannot address a message.
CREATE TABLE IF NOT EXISTS provider_refs (
    event_id             TEXT NOT NULL,
    provider             TEXT NOT NULL,
    provider_message_id  TEXT NOT NULL,
    chat_jid             TEXT,
    from_me              INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(provider, provider_message_id)
);

CREATE TABLE IF NOT EXISTS delivery_events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    occurred_at_ms  INTEGER NOT NULL,
    state           TEXT NOT NULL,
    detail          TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
    event_id UNINDEXED,
    text
);

CREATE TRIGGER IF NOT EXISTS trg_events_fts_insert
AFTER INSERT ON conversation_events
WHEN new.text IS NOT NULL
BEGIN
    INSERT INTO conversation_fts(event_id, text) VALUES (new.id, new.text);
END;
"#;
