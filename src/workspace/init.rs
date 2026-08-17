use crate::config::Config;
use crate::memory::generations::GenerationManager;
use crate::workspace::templates::*;
use anyhow::{Context, Result};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::Command;
use tracing::{info, warn};

pub struct WorkspaceInit;

impl WorkspaceInit {
    pub fn init(config: &Config) -> Result<()> {
        info!(
            "Initializing workspace at {:?} for owner {:?}",
            config.workspace_dir, config.owner_name
        );

        // 1. Create base directories
        let dirs_to_create = vec![
            config.workspace_dir.clone(),
            config.runtime_dir(),
            config.runtime_dir().join("locks"),
            config.runtime_dir().join("tmp"),
            config.runtime_dir().join("media-cache"),
            config.generations_dir(),
            config.generations_dir().join("00000001"),
            config.staging_dir(),
            config.logs_dir(),
            config.workspace_dir.join("history"),
            config.history_jsonl_dir(),
            config.history_assets_dir(),
            config.projects_dir(),
            config.tasks_dir(),
            config.codex_home_dir(),
        ];

        for dir in dirs_to_create {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory {:?}", dir))?;
        }

        // 2. Initial memory generation 00000001
        let gen1_dir = config.generations_dir().join("00000001");
        let index_md = gen1_dir.join("INDEX.md");
        if !index_md.exists() {
            fs::write(
                &index_md,
                "# Active Memory Index\n\n- [USER.md](USER.md): Facts and preferences about the user.\n- [HORIZON.md](HORIZON.md): Current active focus and pending horizons.\n",
            )?;
        }
        let horizon_md = gen1_dir.join("HORIZON.md");
        if !horizon_md.exists() {
            fs::write(
                &horizon_md,
                "# Horizon Context\n\nNo active long-term horizon goals registered.\n",
            )?;
        }
        // Seeded with the configured owner and nothing else. Anything more would
        // be this daemon inventing facts about someone it has not met; the agent
        // fills the rest in from conversation.
        let user_md = gen1_dir.join("USER.md");
        if !user_md.exists() {
            fs::write(
                &user_md,
                format!(
                    "# User profile\n\n- Name: {}\n\nEverything else here is learned from conversation. Do not invent it.\n",
                    config.owner_name
                ),
            )?;
        }

        // 3. Point `MEMORIES` at the newest generation. Not always at 00000001:
        //    re-init on an existing workspace must not roll memory back to the
        //    day it was created.
        Self::remove_legacy_memories_link(config);
        let active = GenerationManager::get_current_generation_num(config)?;
        GenerationManager::point_memories_at(config, active)?;

        // 4. Instruction files.
        //
        //    Ours are refreshed every start so an improved template actually
        //    reaches a live workspace; the user's persona file is written once and
        //    then left alone. See templates.rs for why the split exists.
        Self::write_generated(&config.root_agents_path(), &root_agents_template(config))?;
        Self::write_generated(
            &config.projects_dir().join("AGENTS.md"),
            &projects_agents_template(config),
        )?;
        Self::write_generated(
            &config.tasks_dir().join("AGENTS.md"),
            &tasks_agents_template(config),
        )?;
        Self::write_generated(
            &config.workspace_dir.join("history").join("SCHEMA.md"),
            &history_schema_template(config),
        )?;
        Self::write_generated(
            &config.logs_dir().join("SCHEMA.md"),
            &logs_schema_template(config),
        )?;
        Self::write_generated(&config.workspace_dir.join("WORKING.md"), &working_template(config))?;
        Self::write_generated(
            &config.codex_home_dir().join("AGENTS.md"),
            &codex_bootstrap_template(config),
        )?;
        Self::write_file_if_missing(&config.persona_path(), &persona_template(config))?;
        // The agent's notes on the host. Seeded as a skeleton once, then never
        // touched again, anything it learned about the machine is not recoverable
        // from history and there is nowhere else it could have been written down.
        Self::write_file_if_missing(&config.system_notes_path(), &system_notes_template(config))?;

        // Codex config is regenerated every start: unlike the instruction files
        // it is pure derived state, and it encodes absolute paths that change
        // when the workspace or the binary moves.
        fs::write(
            config.codex_home_dir().join("config.toml"),
            generate_codex_config(config),
        )?;

        // 5. Share the operator's Codex credentials with the workspace home.
        Self::link_codex_credentials(config);

        // 6. Check CLI tools
        Self::check_binary_dependencies();

        info!("Workspace initialization complete!");
        Ok(())
    }

    /// Delete the pre-1.2 lowercase `memories` symlink.
    ///
    /// The link is now `MEMORIES`, matching every other knowledge file here. On
    /// macOS the two names are the same path, so this finds nothing; on Linux a
    /// live workspace would otherwise keep both, and the stale one still resolves , 
    /// which is worse than a broken link, because it silently keeps working while
    /// nothing updates it.
    ///
    /// Only ever removes a symlink. A real directory under that name is somebody's
    /// data and is left alone with a warning.
    fn remove_legacy_memories_link(config: &Config) {
        let legacy = config.legacy_memories_link();
        if legacy == config.memories_link() {
            return;
        }
        match fs::symlink_metadata(&legacy) {
            Err(_) => {}
            Ok(meta) if meta.file_type().is_symlink() => {
                match fs::remove_file(&legacy) {
                    Ok(()) => info!("Removed the superseded {:?} symlink", legacy),
                    Err(e) => warn!("Could not remove {:?}: {e}", legacy),
                }
            }
            Ok(_) => warn!(
                "{:?} exists and is not a symlink; leaving it alone. Active memory is {:?}.",
                legacy,
                config.memories_link()
            ),
        }
    }

    /// Point `<workspace>/.codex-home/auth.json` at the operator's real Codex
    /// credentials.
    ///
    /// A private CODEX_HOME starts with no credentials, so every turn would fail
    /// to authenticate. A symlink (rather than a copy) means refreshed tokens
    /// stay valid for both the daemon and the interactive `codex` CLI, and the
    /// secret is never duplicated onto disk.
    fn link_codex_credentials(config: &Config) {
        let link = config.codex_home_dir().join("auth.json");
        if fs::symlink_metadata(&link).is_ok() {
            return;
        }

        let Some(source) = dirs_home().map(|h| h.join(".codex").join("auth.json")) else {
            warn!("Cannot determine home directory; skipping Codex credential link");
            return;
        };

        if !source.exists() {
            warn!(
                "No Codex credentials at {:?}. Run `codex login` or Codex turns will fail to authenticate.",
                source
            );
            return;
        }

        match symlink(&source, &link) {
            Ok(()) => info!("Linked Codex credentials into {:?}", link),
            Err(e) => warn!("Failed to link Codex credentials into {:?}: {}", link, e),
        }
    }

    fn write_file_if_missing(path: &Path, content: &str) -> Result<()> {
        if !path.exists() {
            fs::write(path, content)
                .with_context(|| format!("Failed to write template file {:?}", path))?;
        }
        Ok(())
    }

    /// Write a machine-owned instruction file, refreshing it in place.
    ///
    /// A file we did not write, no generated marker, is treated as the user's.
    /// It is moved aside rather than destroyed, because instructions someone
    /// hand-wrote are not recoverable from anywhere else.
    fn write_generated(path: &Path, content: &str) -> Result<()> {
        match fs::read_to_string(path) {
            Ok(existing) if existing == content => return Ok(()),
            Ok(existing) if !existing.starts_with(GENERATED_MARKER_PREFIX) => {
                let backup = Self::free_backup_path(path);
                warn!(
                    "{:?} was not written by tera; preserving it at {:?} and installing \
                     the current instructions. Put your own wording in PERSONA.md instead.",
                    path, backup
                );
                fs::rename(path, &backup)
                    .with_context(|| format!("Failed to back up {:?}", path))?;
            }
            Ok(_) => info!("Refreshing generated instructions at {:?}", path),
            Err(_) => {}
        }

        fs::write(path, content)
            .with_context(|| format!("Failed to write generated file {:?}", path))
    }

    /// A backup name that is not already taken.
    ///
    /// Overwriting `AGENTS.md.user-backup` would destroy the very thing the
    /// backup exists to protect: the user's own instructions from the first time
    /// this happened.
    fn free_backup_path(path: &Path) -> std::path::PathBuf {
        let first = path.with_extension("md.user-backup");
        if !first.exists() {
            return first;
        }
        for n in 2..1000 {
            let candidate = path.with_extension(format!("md.user-backup.{n}"));
            if !candidate.exists() {
                return candidate;
            }
        }
        first
    }

    fn check_binary_dependencies() {
        let tools = vec!["git", "sqlite3", "jq", "ffmpeg", "codex"];
        for tool in tools {
            if Command::new(tool).arg("--version").output().is_err() {
                warn!("Optional/required system tool '{}' not found in PATH", tool);
            }
        }
    }
}

/// The operator's home directory. Kept local instead of pulling in a crate for
/// one lookup; the daemon only targets Unix.
fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_init_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);

        WorkspaceInit::init(&config).unwrap();
        assert!(config.root_agents_path().exists());
        assert!(config.persona_path().exists());
        assert!(config.codex_home_dir().join("config.toml").exists());

        let before = fs::read_to_string(config.root_agents_path()).unwrap();
        WorkspaceInit::init(&config).unwrap();
        assert_eq!(fs::read_to_string(config.root_agents_path()).unwrap(), before);
    }

    /// The user's own file is his. Re-init must not touch it.
    #[test]
    fn test_persona_survives_reinit() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);

        WorkspaceInit::init(&config).unwrap();
        fs::write(config.persona_path(), "# Be terse\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert_eq!(
            fs::read_to_string(config.persona_path()).unwrap(),
            "# Be terse\n"
        );
    }

    /// Improved instructions have to reach a workspace that already exists , 
    /// writing them only when absent froze the first generation forever.
    #[test]
    fn test_generated_instructions_are_refreshed() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        // An older generation of our own template.
        fs::write(
            config.root_agents_path(),
            format!("{GENERATED_MARKER_PREFIX} tera -->\n# Ancient instructions\n"),
        )
        .unwrap();
        WorkspaceInit::init(&config).unwrap();

        let refreshed = fs::read_to_string(config.root_agents_path()).unwrap();
        assert!(!refreshed.contains("Ancient instructions"));
        assert!(refreshed.contains("# Operating instructions"));
    }

    /// A file we did not write is not ours to delete.
    #[test]
    fn test_hand_written_instructions_are_backed_up_not_lost() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        fs::write(config.root_agents_path(), "# My own rules\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        let backup = config.root_agents_path().with_extension("md.user-backup");
        assert_eq!(fs::read_to_string(backup).unwrap(), "# My own rules\n");
        assert!(fs::read_to_string(config.root_agents_path())
            .unwrap()
            .starts_with(GENERATED_MARKER_PREFIX));
    }

    /// The marker names the product and says where user edits belong, so its
    /// wording changes, at the rename to tera, it changed for every file. If
    /// detection matched the whole marker, every workspace would have had its own
    /// instructions filed away as hand-written on the next start.
    #[test]
    fn test_an_older_marker_wording_is_still_recognised_as_ours() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        fs::write(
            config.root_agents_path(),
            "<!-- generated: assistantd. Edits are overwritten; put yours in PERSONA.md -->\n# Old\n",
        )
        .unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert!(
            !config
                .root_agents_path()
                .with_extension("md.user-backup")
                .exists(),
            "our own file was mistaken for the user's"
        );
    }

    /// The second time a hand-written file turns up, the first backup must
    /// survive. It is the only copy of what the user actually wrote.
    #[test]
    fn test_a_second_backup_does_not_overwrite_the_first() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        fs::write(config.root_agents_path(), "# First\n").unwrap();
        WorkspaceInit::init(&config).unwrap();
        fs::write(config.root_agents_path(), "# Second\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        let root = config.root_agents_path();
        assert_eq!(
            fs::read_to_string(root.with_extension("md.user-backup")).unwrap(),
            "# First\n"
        );
        assert_eq!(
            fs::read_to_string(root.with_extension("md.user-backup.2")).unwrap(),
            "# Second\n"
        );
    }

    #[test]
    fn test_bootstrap_instructions_point_at_this_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        let bootstrap =
            fs::read_to_string(config.codex_home_dir().join("AGENTS.md")).unwrap();
        assert!(bootstrap.contains(&config.root_agents_path().display().to_string()));
    }
}
