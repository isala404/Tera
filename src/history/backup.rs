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
        [dest_path
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 backup path"))?],
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
    pub projection_valid: bool,
    pub missing_assets: Vec<String>,
}

impl IntegrityReport {
    pub fn is_healthy(&self) -> bool {
        self.sqlite_ok
            && !self.projection_dirty
            && self.projection_valid
            && self.event_count == self.projected_records
            && self.missing_assets.is_empty()
    }
}

pub fn check_integrity(
    config: &Config,
    db: &crate::history::db::HistoryDb,
) -> Result<IntegrityReport> {
    let conn = Connection::open(config.history_db_path())?;
    let result: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;

    let projection_valid = crate::history::projection::ProjectionEngine::validate_projection(
        config.history_jsonl_dir().as_path(),
        db,
    )
    .unwrap_or(false);

    let mut report = IntegrityReport {
        sqlite_ok: result == "ok",
        event_count: db.count_events()?,
        projected_records: crate::history::projection::ProjectionEngine::projected_line_count(
            config.history_jsonl_dir().as_path(),
        )?,
        projection_dirty: crate::history::projection::ProjectionEngine::is_dirty(
            config.history_jsonl_dir().as_path(),
        ),
        projection_valid,
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

/// Remove scratch directories left behind by an interrupted run.
///
/// Scratch is always rebuilt from nothing before use, so anything found here at
/// boot is debris from a crash, and it is debris that takes disk.
pub fn clear_stale_scratch(config: &Config) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    let root = config.runtime_dir().join("tmp");
    if !root.is_dir() {
        return Ok(removed);
    }
    for entry in fs::read_dir(&root)? {
        let path = entry?.path();
        if path.is_dir() {
            fs::remove_dir_all(&path)
                .with_context(|| format!("Failed to remove stale scratch {path:?}"))?;
            removed.push(path);
        }
    }
    Ok(removed)
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
        db.insert_event(ConversationEvent::message(
            "m_1",
            1_786_962_664_000,
            "user",
            Some("remember this".to_string()),
            None,
            None,
            vec![],
        ))
        .unwrap();

        let backup = backup_history(&config, "20260817T120000").unwrap();

        let restored = HistoryDb::open(&backup, &config.history_jsonl_dir()).unwrap();
        assert_eq!(restored.count_events().unwrap(), 1);
        assert_eq!(
            restored.get_event("m_1").unwrap().unwrap().text().unwrap(),
            "remember this"
        );
    }

    #[test]
    fn test_integrity_reports_a_missing_asset() {
        let (_tmp, config) = workspace();
        let db = HistoryDb::open_for(&config).unwrap();
        db.insert_event(ConversationEvent::message(
            "m_2",
            1_786_962_664_000,
            "user",
            None,
            None,
            None,
            vec![Attachment {
                id: None,
                event_id: "m_2".to_string(),
                position: 0,
                media_type: "image".to_string(),
                relative_path: "../assets/2026/08/m_2/gone.jpg".to_string(),
                mime_type: None,
                original_name: Some("gone.jpg".to_string()),
            }],
        ))
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
        db.insert_event(ConversationEvent::message(
            "m_3",
            1_786_962_664_000,
            "assistant",
            Some("fine".to_string()),
            None,
            None,
            vec![],
        ))
        .unwrap();

        assert!(check_integrity(&config, &db).unwrap().is_healthy());
    }

    #[test]
    fn test_stale_scratch_is_cleared() {
        let (_tmp, config) = workspace();
        let debris = config.runtime_dir().join("tmp").join("half-a-download");
        fs::create_dir_all(debris.join("nested")).unwrap();
        fs::write(debris.join("part"), "half-written").unwrap();

        let removed = clear_stale_scratch(&config).unwrap();

        assert!(!debris.exists());
        assert!(removed.iter().any(|p| p.ends_with("half-a-download")));
    }
}
