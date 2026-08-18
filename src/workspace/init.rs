use crate::config::Config;
use crate::memory::generations::GenerationManager;
use crate::workspace::templates::*;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::ffi::CString;
use std::io::{self, ErrorKind};
use std::os::unix::fs::{symlink, PermissionsExt};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use tracing::{info, warn};

#[cfg(target_os = "macos")]
extern "C" {
    fn renamex_np(
        from: *const libc::c_char,
        to: *const libc::c_char,
        flags: libc::c_uint,
    ) -> libc::c_int;
}

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
            config.skills_dir(),
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

        // Native skills are seeded once. Existing paths, edits, symlinks, and
        // deliberate deletions remain user-owned; untouched managed packages can
        // receive a later embedded update.
        Self::seed_builtin_skills(config)?;

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

    /// Seed, update, or remember each bundled skill without taking ownership of
    /// a path the user already has. The state file is outside .agents/skills so
    /// deleting a built-in remains a durable choice.
    fn seed_builtin_skills(config: &Config) -> Result<()> {
        let mut state = Self::load_builtin_skill_state(config)?;

        for skill in crate::data::BUILTIN_SKILLS {
            validate_builtin_skill(skill)?;

            let source_state = embedded_skill_state(skill);
            let source_fingerprint = fingerprint_skill_state(&source_state);
            let destination = config.skills_dir().join(skill.name);

            match state.skills.get(skill.name).cloned() {
                None => {
                    if has_path(&destination) {
                        // The first revision seeded packages before this state
                        // file existed. An untouched package whose files are a
                        // byte-for-byte subset of the new embedded package can
                        // be migrated safely. Anything else is user-owned.
                        if let Some(existing_state) = installed_skill_state(&destination)? {
                            let legacy_managed = !existing_state.files.is_empty()
                                && existing_state.files.iter().all(|(path, fingerprint)| {
                                    source_state.files.get(path) == Some(fingerprint)
                                })
                                && existing_state.files.keys().all(|path| {
                                    existing_state.executables.contains(path)
                                        == source_state.executables.contains(path)
                                });
                            if legacy_managed
                                && Self::update_managed_skill(
                                    &config.skills_dir(),
                                    skill,
                                    &existing_state,
                                )?
                            {
                                info!(
                                    "Migrated untouched built-in skill {:?} at {:?}",
                                    skill.name, destination
                                );
                                state.skills.insert(
                                    skill.name.to_string(),
                                    BuiltinSkillRecord::managed(source_fingerprint, source_state),
                                );
                                continue;
                            }
                        }
                        warn!(
                            "Built-in skill {:?} already exists at {:?}; treating it as user-owned",
                            skill.name, destination
                        );
                        state.skills.insert(
                            skill.name.to_string(),
                            BuiltinSkillRecord::user_owned(),
                        );
                        continue;
                    }

                    if Self::install_new_skill(&config.skills_dir(), skill)? {
                        info!("Seeded built-in skill {:?} at {:?}", skill.name, destination);
                        state.skills.insert(
                            skill.name.to_string(),
                            BuiltinSkillRecord::managed(source_fingerprint, source_state),
                        );
                    } else if has_path(&destination) {
                        state.skills.insert(
                            skill.name.to_string(),
                            BuiltinSkillRecord::user_owned(),
                        );
                    }
                }
                Some(mut record) => match record.status {
                    BuiltinSkillStatus::Deleted | BuiltinSkillStatus::UserOwned => {}
                    BuiltinSkillStatus::Managed => {
                        if !has_path(&destination) {
                            info!(
                                "Built-in skill {:?} was deleted; preserving that choice",
                                skill.name
                            );
                            record.status = BuiltinSkillStatus::Deleted;
                            state.skills.insert(skill.name.to_string(), record);
                            continue;
                        }

                        let Some(installed_state) = installed_skill_state(&destination)? else {
                            warn!(
                                "Built-in skill {:?} changed at {:?}; preserving it as user-owned",
                                skill.name, destination
                            );
                            record.status = BuiltinSkillStatus::UserOwned;
                            state.skills.insert(skill.name.to_string(), record);
                            continue;
                        };

                        // State written before executable modes were tracked can
                        // be upgraded only when its installed modes match this
                        // build. A mismatch may be a deliberate user edit.
                        if record.executables.is_empty() {
                            if installed_state.executables != source_state.executables {
                                warn!(
                                    "Built-in skill {:?} has untracked mode changes at {:?}; preserving it as user-owned",
                                    skill.name, destination
                                );
                                record.status = BuiltinSkillStatus::UserOwned;
                                state.skills.insert(skill.name.to_string(), record);
                                continue;
                            }
                            record.executables = installed_state.executables.clone();
                            record.fingerprint = fingerprint_skill_state(&installed_state);
                        }

                        let expected_state = record.package_state();
                        if installed_state != expected_state {
                            warn!(
                                "Built-in skill {:?} changed at {:?}; preserving it as user-owned",
                                skill.name, destination
                            );
                            record.status = BuiltinSkillStatus::UserOwned;
                            state.skills.insert(skill.name.to_string(), record);
                            continue;
                        }

                        if record.fingerprint != source_fingerprint {
                            if Self::update_managed_skill(
                                &config.skills_dir(),
                                skill,
                                &expected_state,
                            )? {
                                info!("Updated built-in skill {:?} at {:?}", skill.name, destination);
                                record =
                                    BuiltinSkillRecord::managed(source_fingerprint, source_state);
                            } else {
                                warn!(
                                    "Built-in skill {:?} changed while updating; preserving it as user-owned",
                                    skill.name
                                );
                                record.status = BuiltinSkillStatus::UserOwned;
                            }
                        }
                        state.skills.insert(skill.name.to_string(), record);
                    }
                },
            }
        }

        Self::write_builtin_skill_state(config, &state)
    }

    fn install_new_skill(
        skills_dir: &Path,
        skill: &crate::data::BuiltinSkill,
    ) -> Result<bool> {
        let destination = skills_dir.join(skill.name);
        let staging = Self::create_skill_staging(skills_dir, skill.name)?;
        let install = (|| -> Result<bool> {
            write_skill_files(&staging, skill)?;

            // Check again after writing. This prevents a normal concurrent
            // initializer from being replaced if it appeared during staging.
            if has_path(&destination) {
                return Ok(false);
            }
            rename_skill_without_replace(&staging, &destination)
        })();

        let installed = match install {
            Ok(installed) => installed,
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(error);
            }
        };
        if !installed {
            let _ = fs::remove_dir_all(&staging);
        }
        Ok(installed)
    }

    /// Replace a package only after confirming that its files still match the
    /// last embedded version. The old managed directory is kept as a temporary
    /// backup until the new one is in place, so a failed rename can be restored.
    fn update_managed_skill(
        skills_dir: &Path,
        skill: &crate::data::BuiltinSkill,
        expected_state: &SkillPackageState,
    ) -> Result<bool> {
        let destination = skills_dir.join(skill.name);
        let staging = Self::create_skill_staging(skills_dir, skill.name)?;
        let install = (|| -> Result<bool> {
            write_skill_files(&staging, skill)?;
            if installed_skill_state(&destination)?.as_ref() != Some(expected_state) {
                return Ok(false);
            }

            let backup = free_skill_aux_path(skills_dir, skill.name, "backup")?;
            fs::rename(&destination, &backup)
                .with_context(|| format!("failed to stage old built-in skill {destination:?}"))?;
            match fs::rename(&staging, &destination) {
                Ok(()) => {
                    fs::remove_dir_all(&backup)
                        .with_context(|| format!("failed to remove old built-in skill {backup:?}"))?;
                    Ok(true)
                }
                Err(error) => {
                    let _ = fs::rename(&backup, &destination);
                    Err(error).with_context(|| {
                        format!("failed to install updated built-in skill {destination:?}")
                    })
                }
            }
        })();

        match install {
            Ok(updated) => {
                if !updated {
                    let _ = fs::remove_dir_all(&staging);
                }
                Ok(updated)
            }
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                Err(error)
            }
        }
    }

    fn load_builtin_skill_state(config: &Config) -> Result<BuiltinSkillState> {
        match fs::read(config.builtin_skills_state_path()) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("invalid built-in skill state"),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(BuiltinSkillState::default()),
            Err(error) => Err(error).context("failed to read built-in skill state"),
        }
    }

    fn write_builtin_skill_state(config: &Config, state: &BuiltinSkillState) -> Result<()> {
        let path = config.builtin_skills_state_path();
        let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
        let contents = serde_json::to_vec_pretty(state).context("failed to encode built-in skill state")?;
        fs::write(&temporary, contents)
            .with_context(|| format!("failed to write built-in skill state {temporary:?}"))?;
        if let Err(error) = fs::rename(&temporary, &path) {
            let _ = fs::remove_file(&temporary);
            return Err(error).with_context(|| format!("failed to install built-in skill state {path:?}"));
        }
        Ok(())
    }

    fn create_skill_staging(skills_dir: &Path, skill_name: &str) -> Result<PathBuf> {
        for attempt in 0..1000 {
            let suffix = format!(".{}-staging-{}-{}", skill_name, std::process::id(), attempt);
            let path = skills_dir.join(suffix);
            match fs::create_dir(&path) {
                Ok(()) => return Ok(path),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("Failed to create skill staging directory {:?}", path)
                    });
                }
            }
        }
        Err(anyhow!(
            "Could not find a free staging directory for skill {:?}",
            skill_name
        ))
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct BuiltinSkillState {
    skills: BTreeMap<String, BuiltinSkillRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuiltinSkillRecord {
    status: BuiltinSkillStatus,
    fingerprint: String,
    files: BTreeMap<String, String>,
    #[serde(default)]
    executables: BTreeSet<String>,
}

impl BuiltinSkillRecord {
    fn managed(fingerprint: String, package: SkillPackageState) -> Self {
        Self {
            status: BuiltinSkillStatus::Managed,
            fingerprint,
            files: package.files,
            executables: package.executables,
        }
    }

    fn user_owned() -> Self {
        Self {
            status: BuiltinSkillStatus::UserOwned,
            fingerprint: String::new(),
            files: BTreeMap::new(),
            executables: BTreeSet::new(),
        }
    }

    fn package_state(&self) -> SkillPackageState {
        SkillPackageState {
            files: self.files.clone(),
            executables: self.executables.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkillPackageState {
    files: BTreeMap<String, String>,
    executables: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BuiltinSkillStatus {
    Managed,
    UserOwned,
    Deleted,
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

fn embedded_skill_state(skill: &crate::data::BuiltinSkill) -> SkillPackageState {
    SkillPackageState {
        files: skill
            .files
            .iter()
            .map(|file| (file.relative_path.to_string(), fingerprint(file.contents)))
            .collect(),
        executables: skill
            .files
            .iter()
            .filter(|file| file.executable)
            .map(|file| file.relative_path.to_string())
            .collect(),
    }
}

fn fingerprint_skill_state(state: &SkillPackageState) -> String {
    let mut bytes = Vec::new();
    for (path, fingerprint) in &state.files {
        bytes.extend_from_slice(path.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(fingerprint.as_bytes());
        bytes.push(u8::from(state.executables.contains(path)));
        bytes.push(0xff);
    }
    fingerprint(&bytes)
}

/// A stable, dependency-free content fingerprint. It detects accidental or
/// user edits; it is not used as a security boundary.
fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 14_695_981_039_346_656_037u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    format!("{hash:016x}")
}

fn installed_skill_state(path: &Path) -> Result<Option<SkillPackageState>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("failed to inspect {path:?}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Ok(None);
    }

    let mut state = SkillPackageState {
        files: BTreeMap::new(),
        executables: BTreeSet::new(),
    };
    if !collect_installed_state(path, path, &mut state)? {
        return Ok(None);
    }
    Ok(Some(state))
}

fn collect_installed_state(
    root: &Path,
    current: &Path,
    state: &mut SkillPackageState,
) -> Result<bool> {
    for entry in fs::read_dir(current).with_context(|| format!("failed to read {current:?}"))? {
        let path = entry?.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Ok(false);
        }
        if metadata.is_dir() {
            if !collect_installed_state(root, &path, state)? {
                return Ok(false);
            }
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| anyhow!("skill path escaped its root: {path:?}"))?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            if !is_safe_skill_path(&relative) {
                return Ok(false);
            }
            let contents = fs::read(&path)?;
            state.files.insert(relative.clone(), fingerprint(&contents));
            if metadata.permissions().mode() & 0o111 != 0 {
                state.executables.insert(relative);
            }
        } else {
            return Ok(false);
        }
    }
    Ok(true)
}

fn has_path(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

fn free_skill_aux_path(skills_dir: &Path, skill_name: &str, kind: &str) -> Result<PathBuf> {
    for attempt in 0..1000 {
        let path = skills_dir.join(format!(
            ".{skill_name}-{kind}-{}-{attempt}",
            std::process::id()
        ));
        if !has_path(&path) {
            return Ok(path);
        }
    }
    Err(anyhow!("could not find a free {kind} path for skill {skill_name:?}"))
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
fn rename_skill_without_replace(staging: &Path, destination: &Path) -> Result<bool> {
    #[cfg(target_os = "linux")]
    {
        let source = CString::new(staging.as_os_str().as_bytes())
            .context("skill staging path contains a NUL byte")?;
        let target = CString::new(destination.as_os_str().as_bytes())
            .context("skill destination path contains a NUL byte")?;
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if result == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EEXIST) {
            return Ok(false);
        }
        return Err(error).with_context(|| {
            format!(
                "Failed to install built-in skill at {:?} without replacing an existing path",
                destination
            )
        });
    }

    #[cfg(target_os = "macos")]
    {
        const RENAME_EXCL: libc::c_uint = 0x0000_0004;
        let source = CString::new(staging.as_os_str().as_bytes())
            .context("skill staging path contains a NUL byte")?;
        let target = CString::new(destination.as_os_str().as_bytes())
            .context("skill destination path contains a NUL byte")?;
        let result = unsafe { renamex_np(source.as_ptr(), target.as_ptr(), RENAME_EXCL) };
        if result == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EEXIST) {
            return Ok(false);
        }
        return Err(error).with_context(|| {
            format!(
                "Failed to install built-in skill at {:?} without replacing an existing path",
                destination
            )
        });
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        match fs::rename(staging, destination) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(false),
            Err(error) => Err(error)
                .with_context(|| format!("Failed to install built-in skill at {:?}", destination)),
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

    #[test]
    fn test_builtin_skill_is_seeded_atomically_and_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS.first().unwrap();

        WorkspaceInit::init(&config).unwrap();
        let skill = config.skills_dir().join(builtin.name);
        assert!(skill.join("SKILL.md").exists());
        let first_file = builtin.files.first().unwrap();
        let first_path = skill.join(first_file.relative_path);
        let original = fs::read(&first_path).unwrap();

        fs::write(&first_path, b"user-owned\n").unwrap();
        WorkspaceInit::init(&config).unwrap();
        assert_eq!(fs::read(&first_path).unwrap(), b"user-owned\n");
        assert_ne!(original, b"user-owned\n");
    }

    #[test]
    fn test_builtin_skill_executable_mode_is_seeded_and_tracked() {
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
            .skills_dir()
            .join(builtin.name)
            .join(script.relative_path);
        assert_ne!(
            fs::metadata(&installed).unwrap().permissions().mode() & 0o111,
            0
        );

        fs::set_permissions(&installed, fs::Permissions::from_mode(0o644)).unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert_eq!(
            fs::metadata(&installed).unwrap().permissions().mode() & 0o111,
            0
        );
        let state: serde_json::Value =
            serde_json::from_slice(&fs::read(config.builtin_skills_state_path()).unwrap()).unwrap();
        assert_eq!(state["skills"][builtin.name]["status"], "user-owned");
    }

    #[test]
    fn test_untracked_skill_mode_change_is_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS
            .iter()
            .find(|skill| skill.name == "spotify")
            .unwrap();
        let skill_path = config.skills_dir().join(builtin.name);
        fs::create_dir_all(&skill_path).unwrap();
        write_skill_files(&skill_path, builtin).unwrap();
        let script = skill_path.join("scripts/spotify");
        fs::set_permissions(&script, fs::Permissions::from_mode(0o644)).unwrap();

        WorkspaceInit::init(&config).unwrap();

        assert_eq!(
            fs::metadata(&script).unwrap().permissions().mode() & 0o111,
            0
        );
        let state: serde_json::Value =
            serde_json::from_slice(&fs::read(config.builtin_skills_state_path()).unwrap()).unwrap();
        assert_eq!(state["skills"][builtin.name]["status"], "user-owned");
    }

    #[test]
    fn test_skill_fingerprint_includes_executable_mode() {
        let mut state = SkillPackageState {
            files: BTreeMap::from([("scripts/tool".to_string(), "contents".to_string())]),
            executables: BTreeSet::new(),
        };
        let regular = fingerprint_skill_state(&state);
        state.executables.insert("scripts/tool".to_string());

        assert_ne!(fingerprint_skill_state(&state), regular);
    }

    #[test]
    fn test_existing_skill_directory_and_dangling_symlink_are_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS.first().unwrap();
        let skill_path = config.skills_dir().join(builtin.name);
        fs::create_dir_all(&skill_path).unwrap();
        fs::write(skill_path.join("SKILL.md"), "user skill\n").unwrap();
        WorkspaceInit::init(&config).unwrap();
        assert_eq!(
            fs::read_to_string(skill_path.join("SKILL.md")).unwrap(),
            "user skill\n"
        );

        let symlink_tmp = tempfile::tempdir().unwrap();
        let symlink_config = Config::new(symlink_tmp.path().to_path_buf(), true);
        fs::create_dir_all(&symlink_config.skills_dir()).unwrap();
        let symlink_path = symlink_config.skills_dir().join(builtin.name);
        symlink(
            symlink_tmp.path().join("missing-target"),
            &symlink_path,
        )
        .unwrap();
        WorkspaceInit::init(&symlink_config).unwrap();
        assert!(fs::symlink_metadata(symlink_path).unwrap().file_type().is_symlink());
    }

    #[test]
    fn test_deleting_a_managed_skill_is_not_reversed_on_reinit() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().to_path_buf(), true);
        let builtin = crate::data::BUILTIN_SKILLS.first().unwrap();

        WorkspaceInit::init(&config).unwrap();
        fs::remove_dir_all(config.skills_dir().join(builtin.name)).unwrap();
        WorkspaceInit::init(&config).unwrap();
        WorkspaceInit::init(&config).unwrap();

        assert!(!has_path(&config.skills_dir().join(builtin.name)));
        let state: serde_json::Value =
            serde_json::from_slice(&fs::read(config.builtin_skills_state_path()).unwrap()).unwrap();
        assert_eq!(state["skills"][builtin.name]["status"], "deleted");
    }

    #[test]
    fn test_builtin_skill_paths_must_be_simple_relative_paths() {
        assert!(is_safe_skill_path("scripts/control"));
        assert!(!is_safe_skill_path(""));
        assert!(!is_safe_skill_path("../outside"));
        assert!(!is_safe_skill_path("scripts/../outside"));
        assert!(!is_safe_skill_path("/absolute"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_skill_installation_does_not_replace_a_racing_destination() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("staging");
        let destination = tmp.path().join("skill");
        fs::create_dir(&staging).unwrap();
        fs::create_dir(&destination).unwrap();

        assert!(!rename_skill_without_replace(&staging, &destination).unwrap());
        assert!(staging.exists());
        assert!(destination.exists());
    }
}
