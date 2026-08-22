//! One memory pass: stage, let the model work, validate, promote.
//!
//! Rust owns the transaction and the raw material. It does not do the reading.
//! An earlier version scanned event text for the substring "OpenChoreo" and
//! wrote a fixed bullet list, which is not a memory tree. It is a hardcoded
//! guess that no model upgrade could ever improve.
//!
//! The two flavours differ only in where they start and which prompt drives
//! them. Everything around that, the staging directory, the isolated thread,
//! validation and the atomic swap, is identical, so it is written once.

use crate::codex::tier;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::data;
use crate::history::db::HistoryDb;
use crate::memory::generations::GenerationManager;
use crate::runtime::{ActivityTracker, RuntimeDb};
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Where a pass starts, which also decides what it may throw away at the end.
pub enum Start {
    /// Nothing. The model re-derives memory from canonical history, so a better
    /// model organises memory better than the one before it did.
    Empty,
    /// A copy of active memory, so the model edits rather than replaces.
    CopyOfActive,
}

/// One flavour of pass.
pub struct Pass {
    /// Names the staging directory and every log line about this pass.
    pub label: &'static str,
    pub start: Start,
    /// Stays `"true"` until this pass promotes a generation, so an abandoned or
    /// rejected run is retried in the next idle window.
    pub pending_key: &'static str,
    pub prompt: &'static str,
}

/// Triggered when the default model changes underneath us.
pub const REBUILD: Pass = Pass {
    label: "rebuild",
    start: Start::Empty,
    pending_key: "rebuild_pending",
    prompt: data::MEMORY_REBUILD_PROMPT,
};

/// The nightly compaction, which also looks for repeated technical work that
/// might belong in a skill.
pub const NIGHTLY: Pass = Pass {
    label: "optimizer",
    start: Start::CopyOfActive,
    pending_key: "memory_optimizer_retry_pending",
    prompt: data::MEMORY_OPTIMIZER_PROMPT,
};

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A new generation is active.
    Promoted(u64),
    /// Real work arrived; staging was discarded and a retry is pending.
    Interrupted,
    /// The turn failed, or produced something that failed validation. Active
    /// memory is untouched.
    Rejected(String),
}

impl Pass {
    /// Build the tree the model will edit.
    ///
    /// Left in `.memory/staging/` rather than a temp dir so it is on the same
    /// filesystem as the generations, the promotion is a rename, and a rename
    /// across filesystems is not atomic.
    pub fn prepare_staging(&self, config: &Config) -> Result<PathBuf> {
        let staging = config.staging_dir().join(self.label);
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        if let Start::CopyOfActive = self.start {
            let active = config.memories_link();
            if active.exists() {
                copy_tree(&active, &staging)?;
            }
        }

        // A generation with neither of these cannot be validated, and the model
        // needs somewhere to start.
        for (name, seed) in [
            ("INDEX.md", data::MEMORY_INDEX_SEED),
            ("HORIZON.md", data::MEMORY_HORIZON_SEED),
        ] {
            let path = staging.join(name);
            if !path.exists() {
                fs::write(path, seed)?;
            }
        }

        Ok(staging)
    }

    /// Run one pass.
    ///
    /// `activity` is `Some` when something else could need the machine, and the
    /// pass then aborts the moment a conversation turn or scheduled run starts:
    /// the user must never wait behind memory maintenance, and a half-edited
    /// staging tree is not worth salvaging. A one-shot CLI invocation passes
    /// `None`, because the operator asked for this and nothing else is in flight.
    pub async fn run(
        &self,
        config: &Config,
        history_db: &HistoryDb,
        runtime_db: &RuntimeDb,
        codex: &CodexSupervisor,
        activity: Option<&ActivityTracker>,
    ) -> Result<Outcome> {
        let event_count = history_db.count_events()?;
        let staging = self.prepare_staging(config)?;
        info!(
            "Memory {} pass over {event_count} canonical events in {:?}",
            self.label, staging
        );

        let thread_id = codex.start_isolated_thread(&staging, tier::HEAVY).await?;
        let prompt = self.render_prompt(config, &staging, event_count);
        let turn = codex.run_turn_on_thread(&thread_id, &prompt, tier::HEAVY);

        let result = match activity {
            Some(activity) => tokio::select! {
                biased;

                _ = activity.wait_for_work() => {
                    warn!("Real work arrived during the memory {} pass; abandoning it", self.label);
                    let _ = codex.interrupt_thread(&thread_id).await;
                    return self.abandon(runtime_db, &staging, Outcome::Interrupted);
                }

                result = turn => result,
            },
            None => turn.await,
        };

        match result {
            Ok(summary) => info!(
                "Memory {} pass finished: {}",
                self.label,
                summary.chars().take(400).collect::<String>()
            ),
            Err(e) => {
                let reason = format!("memory {} turn failed: {e}", self.label);
                return self.abandon(runtime_db, &staging, Outcome::Rejected(reason));
            }
        }

        // Only safe because staging started empty: anything under these names
        // came out of this turn. A pass that started from a copy could be
        // deleting real memory.
        if let Start::Empty = self.start {
            remove_scratch(&staging);
        }

        // The swap validates the tree before it renames it. Validation is
        // deliberately after the model, not instead of it: Rust checks shape,
        // the model owns content.
        let generation = match GenerationManager::atomic_swap_generation(config, &staging) {
            Ok(generation) => generation,
            Err(e) => return self.abandon(runtime_db, &staging, Outcome::Rejected(e.to_string())),
        };

        runtime_db.set_state_value(self.pending_key, "false")?;
        info!(
            "Memory {} pass complete; generation {generation} is active",
            self.label
        );
        Ok(Outcome::Promoted(generation))
    }

    /// Throw the staging tree away and leave the pass marked pending so the next
    /// idle window retries it.
    fn abandon(&self, runtime_db: &RuntimeDb, staging: &Path, outcome: Outcome) -> Result<Outcome> {
        let _ = fs::remove_dir_all(staging);
        runtime_db.set_state_value(self.pending_key, "true")?;
        Ok(outcome)
    }

    /// Both prompts get the whole variable set. Rendering is plain string
    /// replacement, so a variable a template does not mention costs nothing, and
    /// a template gaining a placeholder the other already uses cannot arrive
    /// unfilled.
    fn render_prompt(&self, config: &Config, staging: &Path, event_count: usize) -> String {
        let history = config.workspace_dir.join("history");
        data::render(
            self.prompt,
            &[
                ("OWNER", &config.owner_name),
                ("WORKSPACE", &config.workspace_dir.display().to_string()),
                ("STAGING", &staging.display().to_string()),
                ("MEMORIES", &config.memories_link().display().to_string()),
                ("HISTORY", &history.display().to_string()),
                ("EVENTS", &event_count.to_string()),
                ("JSONL", &config.history_jsonl_dir().display().to_string()),
                ("SQLITE", &config.history_db_path().display().to_string()),
                ("ASSETS", &config.history_assets_dir().display().to_string()),
                ("SCHEMA", &history.join("SCHEMA.md").display().to_string()),
            ],
        )
    }
}

/// Drop the working files a pass leaves behind. Named conventionally rather than
/// guessed at: anything else is the model's output and stays.
fn remove_scratch(staging: &Path) {
    for name in ["work", "scratch", "candidates", "tmp"] {
        let path = staging.join(name);
        if path.is_dir() {
            if let Err(e) = fs::remove_dir_all(&path) {
                warn!("Could not remove memory scratch {path:?}: {e}");
            }
        }
    }
}

/// Copy a directory tree, following the `MEMORIES` symlink but not any inside it.
fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dest = to.join(entry.file_name());
        let meta = fs::symlink_metadata(&src)?;

        if meta.file_type().is_dir() {
            copy_tree(&src, &dest)?;
        } else if meta.file_type().is_file() {
            fs::copy(&src, &dest)
                .with_context(|| format!("Failed to copy {src:?} into staging"))?;
        } else {
            // Anything else would fail validation on the way out anyway.
            warn!("Skipping {src:?} while staging memory: not a regular file");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::init::WorkspaceInit;

    #[test]
    fn test_the_nightly_pass_starts_as_a_copy_of_active_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();
        fs::write(config.memories_link().join("people.md"), "# Amaya\n").unwrap();

        let staging = NIGHTLY.prepare_staging(&config).unwrap();

        assert_eq!(
            fs::read_to_string(staging.join("people.md")).unwrap(),
            "# Amaya\n"
        );
        assert!(staging.join("INDEX.md").is_file());
        assert!(staging.join("HORIZON.md").is_file());
    }

    /// The point of a rebuild is to re-derive, not to re-edit. If it inherited
    /// the current tree the model would tidy what is there instead of going back
    /// to history.
    #[test]
    fn test_a_rebuild_starts_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();
        fs::write(config.memories_link().join("people.md"), "# Amaya\n").unwrap();

        let staging = REBUILD.prepare_staging(&config).unwrap();

        assert!(!staging.join("people.md").exists());
        assert!(staging.join("INDEX.md").is_file());
        assert!(staging.join("HORIZON.md").is_file());
    }

    /// Staging must be discarded between passes, or an abandoned run's edits
    /// would be promoted by the next one.
    #[test]
    fn test_staging_is_reset_each_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        let staging = NIGHTLY.prepare_staging(&config).unwrap();
        fs::write(staging.join("half-finished.md"), "junk").unwrap();

        let staging = NIGHTLY.prepare_staging(&config).unwrap();
        assert!(!staging.join("half-finished.md").exists());
    }

    /// Each pass gets its own staging directory, so a rebuild and a nightly run
    /// cannot promote each other's work.
    #[test]
    fn test_the_two_passes_do_not_share_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        assert_ne!(
            config.staging_dir().join(REBUILD.label),
            config.staging_dir().join(NIGHTLY.label)
        );
    }

    #[test]
    fn test_the_rebuild_prompt_sends_the_model_to_history_not_to_current_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let staging = config.staging_dir().join(REBUILD.label);

        let prompt = REBUILD.render_prompt(&config, &staging, 4_218);

        assert!(prompt.contains("4218"));
        assert!(prompt.contains(&config.history_db_path().display().to_string()));
        assert!(prompt.contains("Do not read or edit the currently active memory"));
        assert!(prompt.contains("subagents"));
        assert!(
            !prompt.contains("{{"),
            "unfilled placeholder in the rebuild prompt"
        );
    }

    #[test]
    fn test_the_nightly_prompt_points_at_staging_and_forbids_active_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let staging = config.staging_dir().join(NIGHTLY.label);

        let prompt = NIGHTLY.render_prompt(&config, &staging, 0);

        assert!(prompt.contains(&staging.display().to_string()));
        assert!(prompt.contains(&config.history_db_path().display().to_string()));
        assert!(prompt.contains(&config.skills_dir().display().to_string()));
        assert!(prompt.contains("existing skill should improve"));
        assert!(prompt.contains("Do not create or edit skills"));
        assert!(
            !prompt.contains("{{"),
            "unfilled placeholder in the optimizer prompt"
        );
    }

    #[test]
    fn test_scratch_directories_are_removed_but_memory_files_are_not() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("rebuild");
        fs::create_dir_all(staging.join("work")).unwrap();
        fs::create_dir_all(staging.join("people")).unwrap();
        fs::write(staging.join("INDEX.md"), "# Index\n").unwrap();

        remove_scratch(&staging);

        assert!(!staging.join("work").exists());
        assert!(staging.join("people").is_dir());
        assert!(staging.join("INDEX.md").is_file());
    }
}
