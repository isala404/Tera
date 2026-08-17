//! The memory optimizer.
//!
//! Rust owns the transaction. Clone the active generation into staging, run the
//! model against the staging copy, validate the result, swap it in atomically , 
//! and nothing else. Deciding what a memory tree *should say* is the model's job
//! (PLAN.md section 2.2); the previous version did the reorganising itself in
//! Rust with substring matches, which is exactly the intelligence that does not
//! belong here.

use crate::codex::tier;
use crate::codex::CodexSupervisor;
use crate::config::Config;
use crate::memory::generations::GenerationManager;
use crate::runtime::RuntimeDb;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::{info, warn};

/// Set when a run was abandoned so the next idle window retries it.
pub const RETRY_PENDING_KEY: &str = "memory_optimizer_retry_pending";

/// Why an optimizer run ended.
#[derive(Debug, PartialEq, Eq)]
pub enum OptimizerOutcome {
    /// A new generation is active.
    Promoted(u64),
    /// Real work arrived; staging was discarded and a retry is pending.
    Interrupted,
    /// The model ran but produced something that failed validation. Active
    /// memory is untouched.
    Rejected(String),
}

pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Clone active memory into staging so the model has something to edit.
    ///
    /// Returns the staging directory. Left in `.memory/staging/` rather than a
    /// temp dir so it is on the same filesystem as the generations, the promotion
    /// is a rename, and a rename across filesystems is not atomic.
    pub fn prepare_staging(config: &Config) -> Result<std::path::PathBuf> {
        let staging = config.staging_dir().join("optimizer");
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        let active = config.memories_link();
        if active.exists() {
            copy_tree(&active, &staging)?;
        }

        // A generation with neither of these cannot be validated, and the model
        // needs somewhere to start.
        for (file, seed) in [
            ("INDEX.md", "# Memory Index\n"),
            ("HORIZON.md", "# Horizon\n"),
        ] {
            let path = staging.join(file);
            if !path.exists() {
                fs::write(path, seed)?;
            }
        }

        Ok(staging)
    }

    /// Run one optimization pass.
    ///
    /// Aborts the moment a conversation turn or scheduled run starts: the user
    /// must never wait behind memory maintenance, and a half-edited staging tree
    /// is not worth salvaging (PLAN.md section 65).
    pub async fn run(
        config: &Config,
        runtime_db: &RuntimeDb,
        codex: &CodexSupervisor,
        activity: &crate::runtime::ActivityTracker,
    ) -> Result<OptimizerOutcome> {
        let staging = Self::prepare_staging(config)?;
        info!("Optimizing memory in staging copy {:?}", staging);

        let thread_id = codex.start_isolated_thread(&staging, tier::HEAVY).await?;
        let prompt = Self::prompt(config, &staging);

        let outcome = tokio::select! {
            biased;

            _ = activity.wait_for_work() => {
                warn!("Real work arrived during memory optimization; abandoning this pass");
                let _ = codex.interrupt_thread(&thread_id).await;
                let _ = fs::remove_dir_all(&staging);
                runtime_db.set_state_value(RETRY_PENDING_KEY, "true")?;
                return Ok(OptimizerOutcome::Interrupted);
            }

            result = codex.run_turn_on_thread(&thread_id, &prompt, tier::HEAVY) => result,
        };

        match outcome {
            Ok(summary) => info!("Optimizer finished: {}", summary.chars().take(400).collect::<String>()),
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                runtime_db.set_state_value(RETRY_PENDING_KEY, "true")?;
                return Ok(OptimizerOutcome::Rejected(format!("optimizer turn failed: {e}")));
            }
        }

        // Validation is deliberately after the model, not instead of it: Rust
        // checks shape, the model owns content.
        if let Err(e) = GenerationManager::validate_generation_dir(&staging) {
            let _ = fs::remove_dir_all(&staging);
            runtime_db.set_state_value(RETRY_PENDING_KEY, "true")?;
            return Ok(OptimizerOutcome::Rejected(e.to_string()));
        }

        let generation = GenerationManager::atomic_swap_generation(config, &staging)?;
        runtime_db.set_state_value(RETRY_PENDING_KEY, "false")?;
        info!("Memory optimization complete; generation {generation} is active");
        Ok(OptimizerOutcome::Promoted(generation))
    }

    /// The optimizer's instructions (PLAN.md section 63); text in
    /// `data/prompts/memory-optimizer.md`.
    fn prompt(config: &Config, staging: &Path) -> String {
        crate::data::render(
            crate::data::MEMORY_OPTIMIZER_PROMPT,
            &[
                ("OWNER", &config.owner_name),
                ("STAGING", &staging.display().to_string()),
                ("MEMORIES", &config.memories_link().display().to_string()),
                ("HISTORY", &config.workspace_dir.join("history").display().to_string()),
                ("JSONL", &config.history_jsonl_dir().display().to_string()),
                ("SQLITE", &config.history_db_path().display().to_string()),
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
}

/// Copy a directory tree, following the `memories` symlink but not any inside it.
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
    fn test_staging_starts_as_a_copy_of_active_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();
        fs::write(config.memories_link().join("people.md"), "# Amaya\n").unwrap();

        let staging = MemoryOptimizer::prepare_staging(&config).unwrap();

        assert_eq!(
            fs::read_to_string(staging.join("people.md")).unwrap(),
            "# Amaya\n"
        );
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

        let staging = MemoryOptimizer::prepare_staging(&config).unwrap();
        fs::write(staging.join("half-finished.md"), "junk").unwrap();

        let staging = MemoryOptimizer::prepare_staging(&config).unwrap();
        assert!(!staging.join("half-finished.md").exists());
    }

    #[test]
    fn test_prompt_points_at_staging_and_forbids_active_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let staging = config.staging_dir().join("optimizer");

        let prompt = MemoryOptimizer::prompt(&config, &staging);
        assert!(prompt.contains(&staging.display().to_string()));
        assert!(prompt.contains(&config.history_db_path().display().to_string()));
        assert!(prompt.contains("read-only"));
        assert!(prompt.contains("Do not call send_message"));
        assert!(!prompt.contains("{{"), "unfilled placeholder in the optimizer prompt");
    }
}
