//! Full memory regeneration.
//!
//! Different from the nightly optimizer in one way that matters: it starts from
//! an empty generation and rebuilds from canonical history, so a better model
//! organises memory better than the one before it did (PLAN.md sections 47-49).
//!
//! Rust supplies the transaction and the raw material. It does not do the
//! reading: an earlier version scanned event text for the substring "OpenChoreo"
//! and wrote a fixed bullet list, which is not a memory tree. It is a hardcoded
//! guess that no model upgrade could ever improve.

use crate::codex::tier;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::history::db::HistoryDb;
use crate::memory::generations::GenerationManager;
use crate::runtime::RuntimeDb;
use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;
use tracing::{info, warn};

/// Set when a model change means memory should be rebuilt from scratch.
pub const REBUILD_PENDING_KEY: &str = "rebuild_pending";

pub struct MemoryRebuilder;

impl MemoryRebuilder {
    pub async fn run(
        config: &Config,
        history_db: &HistoryDb,
        runtime_db: &RuntimeDb,
        codex: &CodexSupervisor,
    ) -> Result<u64> {
        let event_count = history_db.count_events()?;
        info!("Regenerating memory from {event_count} canonical events");

        let staging = config.staging_dir().join("rebuild");
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        // An empty generation, not a copy of the current one: the point is to
        // re-derive, not to re-edit.
        fs::write(staging.join("INDEX.md"), "# Memory Index\n")?;
        fs::write(staging.join("HORIZON.md"), "# Horizon\n")?;

        let thread_id = codex.start_isolated_thread(&staging, tier::HEAVY).await?;
        let prompt = Self::prompt(config, &staging, event_count);

        match codex.run_turn_on_thread(&thread_id, &prompt, tier::HEAVY).await {
            Ok(summary) => info!(
                "Regeneration finished: {}",
                summary.chars().take(400).collect::<String>()
            ),
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(anyhow!("memory regeneration turn failed: {e}"));
            }
        }

        if let Err(e) = GenerationManager::validate_generation_dir(&staging) {
            let _ = fs::remove_dir_all(&staging);
            return Err(anyhow!("regenerated memory failed validation: {e}"));
        }

        // Intermediate extraction files belong to staging only (PLAN.md 48).
        Self::remove_scratch(&staging);

        let generation = GenerationManager::atomic_swap_generation(config, &staging)?;
        runtime_db.set_state_value(REBUILD_PENDING_KEY, "false")?;
        info!("Memory regeneration complete; generation {generation} is active");
        Ok(generation)
    }

    /// Text in `data/prompts/memory-rebuild.md`.
    fn prompt(config: &Config, staging: &Path, event_count: usize) -> String {
        crate::data::render(
            crate::data::MEMORY_REBUILD_PROMPT,
            &[
                ("OWNER", &config.owner_name),
                ("STAGING", &staging.display().to_string()),
                ("MEMORIES", &config.memories_link().display().to_string()),
                ("HISTORY", &config.workspace_dir.join("history").display().to_string()),
                ("EVENTS", &event_count.to_string()),
                ("JSONL", &config.history_jsonl_dir().display().to_string()),
                ("SQLITE", &config.history_db_path().display().to_string()),
                ("ASSETS", &config.history_assets_dir().display().to_string()),
                (
                    "SCHEMA",
                    &config
                        .workspace_dir
                        .join("history")
                        .join("SCHEMA.md")
                        .display()
                        .to_string(),
                ),
            ],
        )
    }

    /// Drop the working files a regeneration leaves behind. Named conventionally
    /// rather than guessed at: anything else is the model's output and stays.
    fn remove_scratch(staging: &Path) {
        for name in ["work", "scratch", "candidates", "tmp"] {
            let path = staging.join(name);
            if path.is_dir() {
                if let Err(e) = fs::remove_dir_all(&path) {
                    warn!("Could not remove regeneration scratch {path:?}: {e}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_sends_the_model_to_history_not_to_current_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let staging = config.staging_dir().join("rebuild");

        let prompt = MemoryRebuilder::prompt(&config, &staging, 4_218);

        assert!(prompt.contains("4218 events") || prompt.contains("4218"));
        assert!(prompt.contains(&config.history_db_path().display().to_string()));
        assert!(prompt.contains("Do not read or edit the currently active memory"));
        assert!(prompt.contains("subagents"));
        assert!(!prompt.contains("{{"), "unfilled placeholder in the rebuild prompt");
    }

    #[test]
    fn test_scratch_directories_are_removed_but_memory_files_are_not() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("rebuild");
        fs::create_dir_all(staging.join("work")).unwrap();
        fs::create_dir_all(staging.join("people")).unwrap();
        fs::write(staging.join("INDEX.md"), "# Index\n").unwrap();

        MemoryRebuilder::remove_scratch(&staging);

        assert!(!staging.join("work").exists());
        assert!(staging.join("people").is_dir());
        assert!(staging.join("INDEX.md").is_file());
    }
}
