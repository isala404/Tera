//! History backups and integrity checks.
//!
//! SQLite is the only authoritative copy of the conversation, and memory is
//! derived from it, losing it loses everything the assistant knows. Backups go
//! through SQLite rather than copying the file, because a plain copy of a live
//! WAL database can be torn.

use crate::config::Config;
use anyhow::{anyhow, Context, Result};
use chrono::Local;
use rusqlite::Connection;
use std::fs;
use std::path::PathBuf;
use tracing::info;

/// Kept alongside history so a backup travels with the workspace.
fn backup_dir(config: &Config) -> PathBuf {
    config.workspace_dir.join("history").join("backups")
}

/// Snapshot canonical history into `history/backups/`, returning the new file.
///
/// `timestamp` is passed in so the caller controls naming (and tests are not at
/// the mercy of the clock).
pub fn backup_history(config: &Config, timestamp: &str) -> Result<PathBuf> {
    let source = config.history_db_path();
    if !source.exists() {
        return Err(anyhow!("No history database at {source:?} to back up"));
    }

    let dir = backup_dir(config);
    fs::create_dir_all(&dir)?;
    let dest_path = dir.join(format!("history-{timestamp}.sqlite3"));

    if dest_path.exists() {
        return Err(anyhow!("A backup already exists at {dest_path:?}"));
    }

    let src = Connection::open(&source)
        .with_context(|| format!("Cannot open history database {source:?}"))?;

    // VACUUM INTO writes a consistent snapshot of a live database, WAL and all,
    // and compacts it on the way out. A filesystem copy can catch it
    // mid-transaction; this cannot.
    src.execute(
        "VACUUM INTO ?1",
        [dest_path.to_str().ok_or_else(|| anyhow!("non-UTF-8 backup path"))?],
    )
    .with_context(|| format!("Failed to write backup to {dest_path:?}"))?;

    info!("Backed up history to {:?}", dest_path);
    Ok(dest_path)
}

pub fn timestamp_now() -> String {
    Local::now().format("%Y%m%dT%H%M%S").to_string()
}

#[derive(Debug, Default)]
pub struct IntegrityReport {
    pub sqlite_ok: bool,
    pub event_count: usize,
    pub projected_records: usize,
    pub projection_dirty: bool,
    pub missing_assets: Vec<String>,
}

impl IntegrityReport {
    pub fn is_healthy(&self) -> bool {
        self.sqlite_ok
            && !self.projection_dirty
            && self.event_count == self.projected_records
            && self.missing_assets.is_empty()
    }
}

pub fn check_integrity(config: &Config, db: &crate::history::db::HistoryDb) -> Result<IntegrityReport> {
    let conn = Connection::open(config.history_db_path())?;
    let result: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;

    let mut report = IntegrityReport {
        sqlite_ok: result == "ok",
        event_count: db.count_events()?,
        projected_records: crate::history::projection::ProjectionEngine::projected_line_count(
            config.history_jsonl_dir().as_path(),
        )?,
        projection_dirty: crate::history::projection::ProjectionEngine::is_dirty(
            config.history_jsonl_dir().as_path(),
        ),
        missing_assets: Vec::new(),
    };

    // An attachment row whose file is gone is worse than no row: the agent will
    // try to read it and get nothing.
    let mut stmt = conn.prepare("SELECT relative_path FROM attachments")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for path in paths {
        let relative = path?;
        if !config.resolve_asset(&relative).exists() {
            report.missing_assets.push(relative);
        }
    }

    Ok(report)
}

/// Remove staging directories left behind by an interrupted run.
///
/// Staging is always rebuilt from scratch before use, so anything found here at
/// boot is debris from a crash, and it is debris that takes disk.
pub fn clear_stale_staging(config: &Config) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for root in [config.staging_dir(), config.runtime_dir().join("tmp")] {
        if !root.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&root)? {
            let path = entry?.path();
            if path.is_dir() {
                fs::remove_dir_all(&path)
                    .with_context(|| format!("Failed to remove stale staging {path:?}"))?;
                removed.push(path);
            }
        }
    }
    Ok(removed)
}

/// Verify `memories` points at a real generation, and repair it if not.
///
/// Power loss during promotion, or a hand-edited workspace, can leave the symlink
/// dangling, and a dangling `memories/` is an assistant with no memory that
/// reports no error while reading it.
pub fn verify_memories_link(config: &Config) -> Result<bool> {
    let link = config.memories_link();
    if link.join("INDEX.md").is_file() {
        return Ok(true);
    }

    let generation = crate::memory::generations::GenerationManager::get_current_generation_num(config)?;
    let target = config.generations_dir().join(format!("{generation:08}"));
    if !target.is_dir() {
        return Err(anyhow!(
            "memories link is broken and generation {generation} is missing at {target:?}"
        ));
    }

    crate::memory::generations::GenerationManager::point_memories_at(config, generation)?;
    info!("Repaired the memories symlink; it now points at generation {generation}");
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::db::{Attachment, ConversationEvent, HistoryDb};
    use crate::workspace::init::WorkspaceInit;

    fn workspace() -> (tempfile::TempDir, Config) {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();
        (tmp, config)
    }

    #[test]
    fn test_backup_is_a_readable_copy_of_history() {
        let (_tmp, config) = workspace();
        let db = HistoryDb::open_for(&config).unwrap();
        db.insert_event(ConversationEvent {
            seq: None,
            id: "m_1".to_string(),
            occurred_at_ms: 1_786_962_664_000,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: Some("remember this".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        })
        .unwrap();

        let backup = backup_history(&config, "20260817T120000").unwrap();

        let restored = HistoryDb::open(&backup, &config.history_jsonl_dir()).unwrap();
        assert_eq!(restored.count_events().unwrap(), 1);
        assert_eq!(
            restored.get_event("m_1").unwrap().unwrap().text.unwrap(),
            "remember this"
        );
    }

    #[test]
    fn test_integrity_reports_a_missing_asset() {
        let (_tmp, config) = workspace();
        let db = HistoryDb::open_for(&config).unwrap();
        db.insert_event(ConversationEvent {
            seq: None,
            id: "m_2".to_string(),
            occurred_at_ms: 1_786_962_664_000,
            kind: "message".to_string(),
            actor: "user".to_string(),
            text: None,
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![Attachment {
                id: None,
                event_id: "m_2".to_string(),
                position: 0,
                media_type: "image".to_string(),
                relative_path: "../assets/2026/08/m_2/gone.jpg".to_string(),
                mime_type: None,
                original_name: Some("gone.jpg".to_string()),
            }],
        })
        .unwrap();

        let report = check_integrity(&config, &db).unwrap();
        assert!(report.sqlite_ok);
        assert_eq!(report.event_count, 1);
        assert_eq!(report.missing_assets.len(), 1);
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_healthy_history_reports_healthy() {
        let (_tmp, config) = workspace();
        let db = HistoryDb::open_for(&config).unwrap();
        db.insert_event(ConversationEvent {
            seq: None,
            id: "m_3".to_string(),
            occurred_at_ms: 1_786_962_664_000,
            kind: "message".to_string(),
            actor: "assistant".to_string(),
            text: Some("fine".to_string()),
            reply_to_id: None,
            turn_id: None,
            reaction_target_id: None,
            reaction_emoji: None,
            attachments: vec![],
        })
        .unwrap();

        assert!(check_integrity(&config, &db).unwrap().is_healthy());
    }

    #[test]
    fn test_stale_staging_is_cleared() {
        let (_tmp, config) = workspace();
        let debris = config.staging_dir().join("optimizer");
        fs::create_dir_all(debris.join("nested")).unwrap();
        fs::write(debris.join("INDEX.md"), "half-written").unwrap();

        let removed = clear_stale_staging(&config).unwrap();

        assert!(!debris.exists());
        assert!(removed.iter().any(|p| p.ends_with("optimizer")));
    }

    #[test]
    fn test_dangling_memories_link_is_repaired() {
        let (_tmp, config) = workspace();
        fs::remove_file(config.memories_link()).unwrap();
        std::os::unix::fs::symlink("nowhere", config.memories_link()).unwrap();

        assert!(!verify_memories_link(&config).unwrap(), "should have repaired");
        assert!(config.memories_link().join("INDEX.md").is_file());
    }

    #[test]
    fn test_a_healthy_memories_link_is_left_alone() {
        let (_tmp, config) = workspace();
        assert!(verify_memories_link(&config).unwrap());
    }
}
