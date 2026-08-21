//! The JSONL projection of canonical history.
//!
//! This is the file the agent actually reads. It is derived state. SQLite is
//! canonical, but it is the fast path for `jq`, `rg` and Python, so a missing
//! record reads to the agent as a conversation that never happened.
//!
//! Appends are therefore driven from `HistoryDb::insert_event` rather than from
//! each call site: the two call sites that wrote events through the MCP tools
//! never appended, which silently dropped every assistant message and reaction
//! from the projection.

use crate::history::db::{ConversationEvent, HistoryDb};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonlRecord {
    Message(JsonlMessage),
    Reaction(JsonlReaction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlAsset {
    #[serde(rename = "type")]
    pub media_type: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

        if event.kind == "reaction" {
            JsonlRecord::Reaction(JsonlReaction {
                id: event.id.clone(),
                t: t_str,
                from: event.actor.clone(),
                reaction: event.reaction_emoji.clone().unwrap_or_default(),
                to: event.reaction_target_id.clone().unwrap_or_default(),
            })
        } else {
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
                turn: event.turn_id.clone(),
                reply_to: event.reply_to_id.clone(),
                text: event.text.clone(),
                assets,
            })
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

    /// Check the projection against canonical history on daemon start, and
    /// rebuild it if they disagree.
    ///
    /// The dirty marker only catches appends that failed loudly. A projection can
    /// also drift because an older build never appended at all, which is exactly
    /// what happened: 9 projected records against 23 canonical events, so the
    /// agent read a conversation with no assistant in it. Counting is cheap and
    /// self-correcting, so it runs every start rather than on suspicion.
    pub fn verify_and_repair(
        jsonl_dir: &Path,
        staging_root: &Path,
        history_db: &HistoryDb,
    ) -> Result<()> {
        let canonical = history_db.count_events()?;
        let projected = Self::projected_line_count(jsonl_dir)?;
        let dirty = Self::is_dirty(jsonl_dir);

        if !dirty && canonical == projected {
            info!("JSONL projection is in sync with history ({canonical} events)");
            return Ok(());
        }

        warn!(
            "JSONL projection is out of sync (history {canonical} events, projection \
             {projected} records{}); rebuilding it",
            if dirty { ", marked dirty" } else { "" }
        );
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
        ConversationEvent {
            seq: None,
            id: id.to_string(),
            occurred_at_ms: at_ms,
            kind: "message".to_string(),
            actor: actor.to_string(),
            text: Some(text.to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        }
    }

    #[test]
    fn test_absent_fields_are_omitted() {
        // The agent reads these lines with jq; a `"reply_to": null` on every
        // record is noise it has to filter.
        let line = serde_json::to_string(&ProjectionEngine::event_to_record(&message(
            "m_1", "user", 1_786_962_664_000, "hi",
        )))
        .unwrap();
        assert!(!line.contains("reply_to"));
        assert!(!line.contains("assets"));
        assert!(!line.contains("null"));
    }

    #[test]
    fn test_reaction_renders_as_a_reaction_record() {
        let mut ev = message("r_1", "user", 1_786_962_664_000, "");
        ev.kind = "reaction".to_string();
        ev.text = None;
        ev.reaction_emoji = Some("❤️".to_string());
        ev.reaction_target_id = Some("m_1".to_string());

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
}
