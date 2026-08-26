use crate::config::Config;
use crate::data;
use crate::workspace::templates::{self, GENERATED_MARKER_PREFIX};
use anyhow::{bail, Context, Result};
use std::fs;
use std::io::ErrorKind;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Component, Path};
use std::process::Command;
use tracing::{info, warn};

/// Programs tera or its instructions invoke by name. Checked once, at init.
const REQUIRED_BINARIES: &[&str] = &["codex", "git", "python3", "sqlite3", "jq", "ffmpeg"];

pub struct WorkspaceInit;

impl WorkspaceInit {
    pub fn init(config: &Config) -> Result<()> {
        info!(
            "Initializing workspace at {:?} for owner {:?}",
            config.workspace_dir, config.owner_name
        );

        // Before anything is created. A workspace half-built by a run that then
        // fails on a missing program is worse than one that was never started.
        Self::require_binaries(REQUIRED_BINARIES)?;

        let dirs_to_create = vec![
            config.workspace_dir.clone(),
            config.runtime_dir(),
            config.runtime_dir().join("locks"),
            config.runtime_dir().join("tmp"),
            config.runtime_dir().join("media-cache"),
            config.memories_dir(),
            config.logs_dir(),
            config.workspace_dir.join("history"),
            config.history_jsonl_dir(),
            config.history_assets_dir(),
            config.projects_dir(),
            config.tasks_dir(),
            config.codex_home_dir(),
            config.skills_dir(),
        ];

        for dir in dirs_to_create {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory {:?}", dir))?;
        }

        // Seeded with the configured owner and nothing else. Anything more would
        // be this daemon inventing facts about someone it has not met; the agent
        // fills the rest in from conversation.
        let memory_seeds = [
            ("INDEX.md", data::MEMORY_INDEX_SEED),
            ("HORIZON.md", data::MEMORY_HORIZON_SEED),
            ("USER.md", data::MEMORY_USER_SEED),
        ];
        for (filename, seed) in memory_seeds {
            let path = config.memories_dir().join(filename);
            if !path.exists() {
                fs::write(&path, templates::render(seed, config))?;
            }
        }

        Self::init_memory_repo(config)?;

        // Instruction files. Ours are refreshed every start so an improved
        // template actually reaches a live workspace; the user's persona file is
        // written once and then left alone. See templates.rs for why the split
        // exists.
        enum Owned {
            ByUs,
            ByThem,
        }

        let instructions = [
            (
                config.root_agents_path(),
                data::WORKSPACE_AGENTS,
                Owned::ByUs,
            ),
            (
                config.projects_dir().join("AGENTS.md"),
                data::PROJECTS_AGENTS,
                Owned::ByUs,
            ),
            (
                config.tasks_dir().join("AGENTS.md"),
                data::TASKS_AGENTS,
                Owned::ByUs,
            ),
            // The reference the agent is pointed at from AGENTS.md. Generated,
            // because it documents our own storage format. There is nothing here
            // for a user to edit.
            (
                config.workspace_dir.join("history").join("SCHEMA.md"),
                data::HISTORY_SCHEMA,
                Owned::ByUs,
            ),
            // How to read the daemon's own log. Split out of AGENTS.md for the
            // same reason as the history reference: it is needed when something
            // looks broken, not on every turn, and AGENTS.md is read at the start
            // of every session.
            (
                config.logs_dir().join("SCHEMA.md"),
                data::LOGS_SCHEMA,
                Owned::ByUs,
            ),
            // Craft, read before real work rather than every session.
            (
                config.workspace_dir.join("WORKING.md"),
                data::WORKING,
                Owned::ByUs,
            ),
            (
                config.codex_home_dir().join("AGENTS.md"),
                data::CODEX_HOME_AGENTS,
                Owned::ByUs,
            ),
            (config.persona_path(), data::PERSONA, Owned::ByThem),
            // The agent's notes on the host. Seeded as a skeleton once, then never
            // touched again, anything it learned about the machine is not
            // recoverable from history and there is nowhere else it could have
            // been written down.
            (
                config.system_notes_path(),
                data::SYSTEM_NOTES,
                Owned::ByThem,
            ),
        ];

        for (path, template, owned) in instructions {
            let rendered = templates::render(template, config);
            match owned {
                Owned::ByUs => Self::write_generated(&path, &rendered)?,
                Owned::ByThem => Self::write_file_if_missing(&path, &rendered)?,
            }
        }

        Self::seed_builtin_skills(config)?;

        // Model and provider settings are user configuration. Tera passes its
        // own paths and permissions as process overrides when Codex starts.
        Self::seed_codex_config(config)?;

        Self::link_codex_credentials(config);

        info!("Workspace initialization complete!");
        Ok(())
    }

    fn init_memory_repo(config: &Config) -> Result<()> {
        let dir = config.memories_dir();

        if !dir.join(".git").exists() {
            git(&dir, &["init", "--quiet", "--initial-branch", "main"])?;
            info!("Initialized the memory repository at {:?}", dir);
        }

        git(&dir, &["config", "user.name", MEMORY_AUTHOR_NAME])?;
        git(&dir, &["config", "user.email", MEMORY_AUTHOR_EMAIL])?;
        // Memory is prose the agent rewrites wholesale. Left to git's default
        // this would be one merge conflict away from unreadable, and there is no
        // second author to merge with anyway.
        git(&dir, &["config", "merge.ours.driver", "true"])?;

        let status = git_output(&dir, &["status", "--porcelain"])?;
        if status.trim().is_empty() {
            return Ok(());
        }

        git(&dir, &["add", "--all"])?;
        git(&dir, &["commit", "--quiet", "-m", "Seed memory"])?;
        info!("Committed the seeded memory tree");
        Ok(())
    }

    /// Seed the Codex configuration file. Model and provider settings are the
    /// owner's, so tera writes it once and never touches it again.
    fn seed_codex_config(config: &Config) -> Result<()> {
        Self::write_file_if_missing(
            &config.codex_config_path(),
            &templates::generate_codex_config(),
        )
    }

    fn seed_builtin_skills(config: &Config) -> Result<()> {
        // Before anything is removed, so an invalid package cannot leave the
        // directory empty.
        for skill in crate::data::BUILTIN_SKILLS {
            validate_builtin_skill(skill)?;
        }

        let root = config.builtin_skills_dir();
        match fs::remove_dir_all(&root) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to clear {root:?}"));
            }
        }

        for skill in crate::data::BUILTIN_SKILLS {
            let destination = root.join(skill.name);
            fs::create_dir_all(&destination)
                .with_context(|| format!("failed to create {destination:?}"))?;
            write_skill_files(&destination, skill)?;
        }

        info!(
            "Wrote {} built-in skills to {:?}",
            crate::data::BUILTIN_SKILLS.len(),
            root
        );
        Ok(())
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

    /// Refuse to initialize without the programs the workspace is built on.
    ///
    /// These are not optional. `codex` runs every turn, `git` is the memory
    /// store, and the instructions hand the agent `sqlite3`, `jq`, `python3` and
    /// `ffmpeg` by name. A missing one used to be a startup warning nobody read,
    /// and then a turn that failed hours later with an error about something
    /// else. Failing here names the real problem once.
    fn require_binaries(required: &[&str]) -> Result<()> {
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|tool| crate::runtime::executable_on_path(tool).is_err())
            .collect();

        if missing.is_empty() {
            return Ok(());
        }

        bail!(
            "tera needs these programs on PATH and cannot find them: {}. \
             Install them and run tera again.",
            missing.join(", ")
        )
    }
}

/// Who commits to the memory repository. Set per-repository, never globally.
const MEMORY_AUTHOR_NAME: &str = "Tera";
const MEMORY_AUTHOR_EMAIL: &str = "tera@localhost";

fn git(dir: &Path, args: &[&str]) -> Result<()> {
    git_output(dir, args).map(|_| ())
}

fn git_output(dir: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .with_context(|| format!("could not run git {}", args.join(" ")))?;

    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn validate_builtin_skill(skill: &crate::data::BuiltinSkill) -> Result<()> {
    if !is_safe_skill_path(skill.name) {
        bail!("invalid built-in skill name {:?}", skill.name);
    }
    if skill.files.is_empty() {
        bail!("built-in skill {:?} has no files", skill.name);
    }
    for file in skill.files {
        if !is_safe_skill_path(file.relative_path) {
            bail!(
                "invalid path {:?} in built-in skill {:?}",
                file.relative_path,
                skill.name
            );
        }
    }
    Ok(())
}

fn write_skill_files(staging: &Path, skill: &crate::data::BuiltinSkill) -> Result<()> {
    for file in skill.files {
        let path = staging.join(file.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create skill directory {parent:?}"))?;
        }
        fs::write(&path, file.contents)
            .with_context(|| format!("failed to write built-in skill file {path:?}"))?;
        let mode = if file.executable { 0o755 } else { 0o644 };
        fs::set_permissions(&path, fs::Permissions::from_mode(mode))
            .with_context(|| format!("failed to set permissions on {path:?}"))?;
    }
    Ok(())
}

fn is_safe_skill_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Move a fully written package into place without replacing a user path.
/// Linux has a native no-replace rename. Other supported Unix systems retain
/// the final existence check and report an existing destination as a skip.
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
    fn test_a_missing_program_stops_initialization_and_names_it() {
        let error = WorkspaceInit::require_binaries(&["git", "tera-no-such-program"]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("tera-no-such-program"), "{message}");
        // Only the missing one. A list that also names what is installed sends
        // the owner looking in the wrong place.
        assert!(!message.contains("git"), "{message}");
    }

    /// Every program the daemon and the shipped instructions invoke by name. If
    /// one of these is not really required, it does not belong in the list.
    #[test]
    fn test_the_required_programs_are_present_here() {
        WorkspaceInit::require_binaries(REQUIRED_BINARIES).unwrap();
    }

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
        assert_eq!(
            fs::read_to_string(config.root_agents_path()).unwrap(),
            before
        );
    }

    #[test]
    fn test_codex_model_config_survives_reinit() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        let custom = "model = \"gpt-5.6-terra\"\nmodel_reasoning_effort = \"medium\"\n";
        fs::write(config.codex_config_path(), custom).unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert_eq!(
            fs::read_to_string(config.codex_config_path()).unwrap(),
            custom
        );
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

        let bootstrap = fs::read_to_string(config.codex_home_dir().join("AGENTS.md")).unwrap();
        assert!(bootstrap.contains(&config.root_agents_path().display().to_string()));
    }

    #[test]
    fn test_builtin_skills_land_where_codex_finds_them() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        for builtin in crate::data::BUILTIN_SKILLS {
            let skill = config.builtin_skills_dir().join(builtin.name);
            assert!(skill.join("SKILL.md").exists(), "{:?}", skill);
        }
        // Codex reads CODEX_HOME/skills recursively, so this is the path it
        // scans. A test that only checked the directory tera writes would pass
        // with the skills somewhere codex never looks.
        assert!(config
            .builtin_skills_dir()
            .starts_with(config.codex_home_dir().join("skills")));
    }

    #[test]
    fn test_an_edited_builtin_skill_is_rewritten() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS.first().unwrap();
        WorkspaceInit::init(&config).unwrap();

        let file = config
            .builtin_skills_dir()
            .join(builtin.name)
            .join(builtin.files.first().unwrap().relative_path);
        let shipped = fs::read(&file).unwrap();
        fs::write(&file, b"edited\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert_eq!(fs::read(&file).unwrap(), shipped);
    }

    #[test]
    fn test_a_skill_tera_no_longer_ships_is_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        let stale = config.builtin_skills_dir().join("retired");
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("SKILL.md"), "old\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert!(!stale.exists());
    }

    #[test]
    fn test_executable_mode_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS
            .iter()
            .find(|skill| skill.name == "spotify")
            .unwrap();
        let script = builtin
            .files
            .iter()
            .find(|file| file.relative_path == "scripts/spotify")
            .unwrap();
        assert!(script.executable);

        WorkspaceInit::init(&config).unwrap();

        let installed = config
            .builtin_skills_dir()
            .join(builtin.name)
            .join(script.relative_path);
        assert_ne!(
            fs::metadata(&installed).unwrap().permissions().mode() & 0o111,
            0
        );
    }

    #[test]
    fn test_the_owners_own_skills_are_left_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        WorkspaceInit::init(&config).unwrap();

        let mine = config.skills_dir().join("mine");
        fs::create_dir_all(&mine).unwrap();
        fs::write(mine.join("SKILL.md"), "mine\n").unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert_eq!(fs::read_to_string(mine.join("SKILL.md")).unwrap(), "mine\n");
    }

    #[test]
    fn test_builtin_skill_paths_must_be_simple_relative_paths() {
        assert!(is_safe_skill_path("scripts/control"));
        assert!(!is_safe_skill_path(""));
        assert!(!is_safe_skill_path("../outside"));
        assert!(!is_safe_skill_path("scripts/../outside"));
        assert!(!is_safe_skill_path("/absolute"));
    }
}
