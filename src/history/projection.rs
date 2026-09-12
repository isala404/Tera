//! The JSONL projection of canonical history.
//!
//! This is the file the agent actually reads. It is derived state. SQLite is
//! canonical, but it is the fast path for `jq`, `rg` and Python, so a missing
//! record reads to the agent as a conversation that never happened.
//!
//! Appends are driven from `HistoryDb::insert_event` rather than from each call
//! site, so there is no way to write an event and forget the projection.

use crate::history::db::{ConversationEvent, EventPayload, HistoryDb};
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Marker left behind when an append fails, so the next daemon start knows the
/// projection has drifted from SQLite and rebuilds it.
const DIRTY_MARKER: &str = ".projection-dirty";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonlRecord {
    Message(JsonlMessage),
    Reaction(JsonlReaction),
}

impl JsonlRecord {
    pub fn id(&self) -> &str {
        match self {
            Self::Message(m) => &m.id,
            Self::Reaction(r) => &r.id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlAsset {
    #[serde(rename = "type")]
    pub media_type: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlMessage {
    pub id: String,
    pub t: String,
    pub from: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assets: Option<Vec<JsonlAsset>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlReaction {
    pub id: String,
    pub t: String,
    pub from: String,
    pub reaction: String,
    pub to: String,
}

pub struct ProjectionEngine;

impl ProjectionEngine {
    fn event_to_record(event: &ConversationEvent) -> JsonlRecord {
        let dt: DateTime<Utc> = Utc.timestamp_millis_opt(event.occurred_at_ms).unwrap();
        let t_str = dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        match &event.payload {
            EventPayload::Reaction { target_id, emoji } => JsonlRecord::Reaction(JsonlReaction {
                id: event.id.clone(),
                t: t_str,
                from: event.actor.clone(),
                reaction: emoji.clone(),
                to: target_id.clone(),
            }),
            EventPayload::Message {
                text,
                reply_to_id,
                turn_id,
            } => {
                let assets = if !event.attachments.is_empty() {
                    Some(
                        event
                            .attachments
                            .iter()
                            .map(|a| JsonlAsset {
                                media_type: a.media_type.clone(),
                                path: a.relative_path.clone(),
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                JsonlRecord::Message(JsonlMessage {
                    id: event.id.clone(),
                    t: t_str,
                    from: event.actor.clone(),
                    turn: turn_id.clone(),
                    reply_to: reply_to_id.clone(),
                    text: text.clone(),
                    assets,
                })
            }
        }
    }

    fn month_file(jsonl_dir: &Path, occurred_at_ms: i64) -> PathBuf {
        let dt: DateTime<Utc> = Utc.timestamp_millis_opt(occurred_at_ms).unwrap();
        jsonl_dir.join(format!("{}.jsonl", dt.format("%Y-%m")))
    }

    pub fn append_event(jsonl_dir: &Path, event: &ConversationEvent) -> Result<()> {
        let jsonl_path = Self::month_file(jsonl_dir, event.occurred_at_ms);

        let record = Self::event_to_record(event);
        let json_line = serde_json::to_string(&record)? + "\n";

        fs::create_dir_all(jsonl_dir)
            .with_context(|| format!("Failed to create JSONL directory {:?}", jsonl_dir))?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
            .with_context(|| format!("Failed to open JSONL projection file {:?}", jsonl_path))?;

        file.write_all(json_line.as_bytes())?;
        file.sync_data()?;
        Ok(())
    }

    /// Record that the projection no longer matches SQLite.
    ///
    /// The canonical event is already committed, so a failed append must not fail
    /// the turn. It just means the projection needs regenerating.
    pub fn mark_dirty(jsonl_dir: &Path) {
        let _ = fs::create_dir_all(jsonl_dir);
        if let Err(e) = fs::write(jsonl_dir.join(DIRTY_MARKER), "projection append failed\n") {
            warn!("Could not write projection dirty marker: {e}");
        }
    }

    pub fn is_dirty(jsonl_dir: &Path) -> bool {
        jsonl_dir.join(DIRTY_MARKER).exists()
    }

    pub fn projected_line_count(jsonl_dir: &Path) -> Result<usize> {
        if !jsonl_dir.exists() {
            return Ok(0);
        }
        let mut total = 0;
        for entry in fs::read_dir(jsonl_dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "jsonl") {
                total += fs::read_to_string(&path)?
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .count();
            }
        }
        Ok(total)
    }

    /// Validate the JSONL projection files against canonical history.
    ///
    /// Checks:
    /// - Not marked dirty.
    /// - Every line parses as valid JSON matching `JsonlRecord`.
    /// - No duplicate IDs.
    /// - Total record count matches canonical event count.
    /// - Every canonical event corresponds to a projected record with identical
    ///   fields (id, timestamp, actor, text/reaction, assets).
    pub fn validate_projection(jsonl_dir: &Path, history_db: &HistoryDb) -> Result<bool> {
        if Self::is_dirty(jsonl_dir) {
            return Ok(false);
        }
        if !jsonl_dir.exists() {
            let count = history_db.count_events()?;
            return Ok(count == 0);
        }

        let mut projected_records = HashMap::new();
        for entry in fs::read_dir(jsonl_dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "jsonl") {
                let content = fs::read_to_string(&path)?;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let record: JsonlRecord = match serde_json::from_str(trimmed) {
                        Ok(r) => r,
                        Err(_) => return Ok(false), // invalid JSON
                    };
                    if projected_records
                        .insert(record.id().to_string(), record)
                        .is_some()
                    {
                        return Ok(false); // duplicate ID
                    }
                }
            }
        }

        let canonical_events = history_db.list_events_all()?;
        if canonical_events.len() != projected_records.len() {
            return Ok(false); // count mismatch
        }

        for event in canonical_events {
            let Some(projected) = projected_records.get(&event.id) else {
                return Ok(false); // missing record
            };
            let expected = Self::event_to_record(&event);
            if projected != &expected {
                return Ok(false); // changed text, mismatched reaction, or differing fields
            }
        }

        Ok(true)
    }

    /// Check the projection against canonical history on daemon start, and
    /// rebuild it if they disagree.
    pub fn verify_and_repair(
        jsonl_dir: &Path,
        staging_root: &Path,
        history_db: &HistoryDb,
    ) -> Result<()> {
        let is_valid = match Self::validate_projection(jsonl_dir, history_db) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to validate projection: {e}; rebuilding");
                false
            }
        };

        if is_valid {
            info!("JSONL projection is in sync with canonical history");
            return Ok(());
        }

        warn!("JSONL projection is out of sync or corrupt; rebuilding it");
        Self::rebuild_all(jsonl_dir, staging_root, history_db)
    }

    pub fn rebuild_all(
        jsonl_dir: &Path,
        staging_root: &Path,
        history_db: &HistoryDb,
    ) -> Result<()> {
        info!("Starting full JSONL projection rebuild...");
        let events = history_db.list_events_all()?;

        let staging_dir = staging_root.join("jsonl_rebuild_staging");
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir)?;
        }
        fs::create_dir_all(&staging_dir)?;

        let mut file_handles: HashMap<String, File> = HashMap::new();

        for event in events {
            let dt: DateTime<Utc> = Utc.timestamp_millis_opt(event.occurred_at_ms).unwrap();
            let month_key = dt.format("%Y-%m").to_string();
            let month_filename = format!("{}.jsonl", month_key);

            let record = Self::event_to_record(&event);
            let line = serde_json::to_string(&record)? + "\n";

            let file = file_handles.entry(month_key).or_insert_with(|| {
                let p = staging_dir.join(&month_filename);
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(p)
                    .unwrap()
            });

            file.write_all(line.as_bytes())?;
        }

        for file in file_handles.values_mut() {
            file.sync_all()?;
        }

        fs::create_dir_all(jsonl_dir)?;

        let mut written = Vec::new();
        for entry in fs::read_dir(&staging_dir)? {
            let entry = entry?;
            written.push(entry.file_name());
            fs::rename(entry.path(), jsonl_dir.join(entry.file_name()))?;
        }

        // A month the rebuild did not produce has no events left in SQLite, so
        // leaving its old file behind would make the projection claim history
        // that the canonical store does not have.
        for entry in fs::read_dir(jsonl_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let is_month_file = name.to_string_lossy().ends_with(".jsonl");
            if is_month_file && !written.contains(&name) {
                info!("Removing stale projection file {:?}", name);
                fs::remove_file(entry.path())?;
            }
        }

        fs::remove_dir_all(&staging_dir)?;
        let _ = fs::remove_file(jsonl_dir.join(DIRTY_MARKER));
        info!("JSONL projection rebuild completed successfully!");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::db::Attachment;

    fn message(id: &str, actor: &str, at_ms: i64, text: &str) -> ConversationEvent {
        ConversationEvent::message(id, at_ms, actor, Some(text.to_string()), None, None, vec![])
    }

    #[test]
    fn test_absent_fields_are_omitted() {
        // The agent reads these lines with jq; a `"reply_to": null` on every
        // record is noise it has to filter.
        let line = serde_json::to_string(&ProjectionEngine::event_to_record(&message(
            "m_1",
            "user",
            1_786_962_664_000,
            "hi",
        )))
        .unwrap();
        assert!(!line.contains("reply_to"));
        assert!(!line.contains("assets"));
        assert!(!line.contains("null"));
    }

    #[test]
    fn test_reaction_renders_as_a_reaction_record() {
        let ev = ConversationEvent::reaction("r_1", 1_786_962_664_000, "user", "m_1", "❤️");

        let v: serde_json::Value =
            serde_json::to_value(ProjectionEngine::event_to_record(&ev)).unwrap();
        assert_eq!(v["reaction"], "❤️");
        assert_eq!(v["to"], "m_1");
        assert!(v.get("text").is_none());
    }

    #[test]
    fn test_attachments_are_listed_with_their_paths() {
        let mut ev = message("m_2", "user", 1_786_962_664_000, "look");
        ev.attachments = vec![Attachment {
            id: None,
            event_id: "m_2".to_string(),
            position: 0,
            media_type: "image".to_string(),
            relative_path: "../assets/2026/08/m_2/photo.jpg".to_string(),
            mime_type: Some("image/jpeg".to_string()),
            original_name: Some("photo.jpg".to_string()),
        }];

        let v: serde_json::Value =
            serde_json::to_value(ProjectionEngine::event_to_record(&ev)).unwrap();
        assert_eq!(v["assets"][0]["type"], "image");
        assert_eq!(v["assets"][0]["path"], "../assets/2026/08/m_2/photo.jpg");
    }

    #[test]
    fn test_events_land_in_their_own_month_file() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("jsonl");

        // 2026-07-31T23:00:00Z and 2026-08-01T01:00:00Z
        ProjectionEngine::append_event(&jsonl, &message("m_1", "user", 1_785_538_800_000, "july"))
            .unwrap();
        ProjectionEngine::append_event(&jsonl, &message("m_2", "user", 1_785_546_000_000, "aug"))
            .unwrap();

        assert!(jsonl.join("2026-07.jsonl").exists());
        assert!(jsonl.join("2026-08.jsonl").exists());
    }

    #[test]
    fn test_dirty_marker_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("jsonl");
        assert!(!ProjectionEngine::is_dirty(&jsonl));
        ProjectionEngine::mark_dirty(&jsonl);
        assert!(ProjectionEngine::is_dirty(&jsonl));
    }

    #[test]
    fn test_validate_projection_detects_corruption_and_repairs() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("history.sqlite3");
        let jsonl_dir = dir.path().join("jsonl");
        let staging = dir.path().join("staging");
        let db = HistoryDb::open(&db_path, &jsonl_dir).unwrap();

        let ev = message("m_1", "user", 1_785_538_800_000, "original message");
        db.insert_event(ev).unwrap();

        // Initially valid
        assert!(ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());

        // 1. Invalid JSON corrupts projection
        let month_file = jsonl_dir.join("2026-07.jsonl");
        fs::write(&month_file, "not JSON at all\n").unwrap();
        assert!(!ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());
        // Verify and repair fixes it
        ProjectionEngine::verify_and_repair(&jsonl_dir, &staging, &db).unwrap();
        assert!(ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());

        // 2. Changed text corrupts projection
        fs::write(
            &month_file,
            "{\"id\":\"m_1\",\"t\":\"2026-07-31T23:00:00.000Z\",\"from\":\"user\",\"text\":\"tampered message\"}\n",
        )
        .unwrap();
        assert!(!ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());
        ProjectionEngine::verify_and_repair(&jsonl_dir, &staging, &db).unwrap();
        assert!(ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());

        // 3. Duplicate IDs corrupt projection
        let rec = serde_json::to_string(&ProjectionEngine::event_to_record(
            &db.get_event("m_1").unwrap().unwrap(),
        ))
        .unwrap();
        fs::write(&month_file, format!("{rec}\n{rec}\n")).unwrap();
        assert!(!ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());
        ProjectionEngine::verify_and_repair(&jsonl_dir, &staging, &db).unwrap();
        assert!(ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());

        // 4. Missing record corrupts projection
        fs::write(&month_file, "").unwrap();
        assert!(!ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());
        ProjectionEngine::verify_and_repair(&jsonl_dir, &staging, &db).unwrap();
        assert!(ProjectionEngine::validate_projection(&jsonl_dir, &db).unwrap());
    }
}
