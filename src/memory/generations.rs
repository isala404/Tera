use crate::config::Config;
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use tracing::info;

/// `HORIZON.md` is meant to be glanceable. Past this it has stopped being a
/// horizon and started being a second memory tree.
const MAX_HORIZON_BYTES: u64 = 32 * 1024;

/// No single memory file should approach this. One that does is usually a
/// history dump that got pasted into memory.
const MAX_MEMORY_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Generous bound on a whole generation. Memory is an interpretation of history,
/// so it has no business being the same order of size as history.
const MAX_GENERATION_BYTES: u64 = 64 * 1024 * 1024;

/// How many generations to keep for rollback.
const GENERATIONS_TO_KEEP: usize = 14;

/// Every generation directory, numbered, oldest first. A name that is not a
/// number is not one of ours, so it is neither counted nor pruned.
fn numbered_generations(config: &Config) -> Result<Vec<(u64, PathBuf)>> {
    let dir = config.generations_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut found: Vec<(u64, PathBuf)> = fs::read_dir(&dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let num = entry.file_name().to_str()?.parse::<u64>().ok()?;
            Some((num, entry.path()))
        })
        .collect();
    found.sort_by_key(|(num, _)| *num);
    Ok(found)
}

pub struct GenerationManager;

impl GenerationManager {
    pub fn get_current_generation_num(config: &Config) -> Result<u64> {
        Ok(numbered_generations(config)?
            .last()
            .map_or(1, |(num, _)| *num))
    }

    /// The generation actually in use, read from the symlink.
    ///
    /// Not the same as the highest number after a rollback, reporting the newest
    /// as "active" would tell the operator the rollback did not take.
    pub fn active_generation(config: &Config) -> Option<u64> {
        let target = fs::read_link(config.memories_link()).ok()?;
        target
            .file_name()?
            .to_str()?
            .parse::<u64>()
            .ok()
    }

    pub fn prepare_next_generation_dir(config: &Config) -> Result<(u64, PathBuf)> {
        let current_num = Self::get_current_generation_num(config)?;
        let next_num = current_num + 1;
        let folder_name = format!("{:08}", next_num);
        let next_dir = config.generations_dir().join(folder_name);
        Ok((next_num, next_dir))
    }

    /// Deterministic checks before a generation may become active.
    ///
    /// These are all shape, never truth: whether the memory is *correct* is a
    /// model and history problem, not something Rust can judge. What Rust can do
    /// is refuse to activate a generation that would break the workspace, a
    /// symlink pointing out of it, a device node, a file that swallowed a history
    /// dump.
    pub fn validate_generation_dir(dir: &Path) -> Result<()> {
        if !dir.is_dir() {
            return Err(anyhow!("Memory validation failed: {dir:?} is not a directory"));
        }

        for required in ["INDEX.md", "HORIZON.md"] {
            let path = dir.join(required);
            if !path.is_file() {
                return Err(anyhow!(
                    "Memory validation failed: missing required file {path:?}"
                ));
            }
        }

        let horizon_bytes = fs::metadata(dir.join("HORIZON.md"))?.len();
        if horizon_bytes > MAX_HORIZON_BYTES {
            return Err(anyhow!(
                "Memory validation failed: HORIZON.md is {horizon_bytes} bytes; it is meant to be \
                 a short peripheral-awareness file (limit {MAX_HORIZON_BYTES})"
            ));
        }

        let mut total_bytes = 0u64;
        Self::walk_generation(dir, &mut total_bytes)?;

        if total_bytes > MAX_GENERATION_BYTES {
            return Err(anyhow!(
                "Memory validation failed: generation is {total_bytes} bytes, over the \
                 {MAX_GENERATION_BYTES} byte safety bound. Memory is derived from history, not a \
                 copy of it"
            ));
        }

        Ok(())
    }

    /// Recursively check every entry is a plain file or directory, summing sizes.
    fn walk_generation(dir: &Path, total_bytes: &mut u64) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            // symlink_metadata, not metadata: a symlink out of the generation
            // would otherwise be validated as whatever it points at, and then
            // promoted, which is how memory ends up "containing" /etc.
            let meta = fs::symlink_metadata(&path)?;
            let file_type = meta.file_type();

            if file_type.is_symlink() {
                return Err(anyhow!(
                    "Memory validation failed: {path:?} is a symlink; a generation must be \
                     self-contained"
                ));
            }

            if file_type.is_dir() {
                Self::walk_generation(&path, total_bytes)?;
                continue;
            }

            if !file_type.is_file() {
                return Err(anyhow!(
                    "Memory validation failed: {path:?} is not a regular file"
                ));
            }

            if meta.len() > MAX_MEMORY_FILE_BYTES {
                return Err(anyhow!(
                    "Memory validation failed: {path:?} is {} bytes, over the \
                     {MAX_MEMORY_FILE_BYTES} byte per-file bound",
                    meta.len()
                ));
            }

            // Readability is part of the contract; an unreadable memory file is
            // indistinguishable from a missing one at read time.
            fs::File::open(&path)
                .with_context(|| format!("Memory validation failed: cannot read {path:?}"))?;

            *total_bytes += meta.len();
        }
        Ok(())
    }

    pub fn atomic_swap_generation(config: &Config, staging_dir: &Path) -> Result<u64> {
        Self::validate_generation_dir(staging_dir)?;

        let (next_num, next_dir) = Self::prepare_next_generation_dir(config)?;

        info!(
            "Promoting staging memory to generation {:08} at {:?}",
            next_num, next_dir
        );

        fs::rename(staging_dir, &next_dir)
            .with_context(|| format!("Failed to move staging memory into {:?}", next_dir))?;

        Self::point_memories_at(config, next_num)?;

        Self::prune_old_generations(config, GENERATIONS_TO_KEEP)?;
        Ok(next_num)
    }

    /// Repoint `memories` at a generation without ever leaving it dangling.
    ///
    /// Removing the old symlink and then creating the new one leaves a window in
    /// which `memories/` does not exist, and a turn reading `memories/INDEX.md`
    /// in that window sees an assistant with no memory at all. Creating the link
    /// under a temporary name and renaming it over the old one is atomic.
    pub fn point_memories_at(config: &Config, generation: u64) -> Result<()> {
        let memories_link = config.memories_link();
        let staging_link = config.workspace_dir.join("memories.new");
        let rel_target = Path::new(".memory")
            .join("generations")
            .join(format!("{generation:08}"));

        let _ = fs::remove_file(&staging_link);
        symlink(&rel_target, &staging_link)
            .with_context(|| format!("Failed to create memories symlink to {rel_target:?}"))?;

        fs::rename(&staging_link, &memories_link).with_context(|| {
            format!("Failed to move {staging_link:?} over {memories_link:?}")
        })?;
        Ok(())
    }

    pub fn prune_old_generations(config: &Config, keep_count: usize) -> Result<()> {
        let generations = numbered_generations(config)?;
        let excess = generations.len().saturating_sub(keep_count);
        for (num, path) in generations.iter().take(excess) {
            info!("Pruning old memory generation {num:08}");
            let _ = fs::remove_dir_all(path);
        }
        Ok(())
    }
}
