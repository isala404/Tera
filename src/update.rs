//! Transactional updates for the daemon and the Codex CLI it owns.
//!
//! The running executable can be replaced safely on Unix, but a successful
//! rename does not prove the replacement can start. The updater therefore keeps
//! the old binary and a small journal until the next daemon reaches its healthy
//! point. Phoenix rolls the binary back after a failed first start.

use crate::config::Config;
use crate::version::{codex_version, BuildInfo};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use tracing::{info, warn};

const RELEASE_BASE_URL: &str = "https://github.com/isala404/Tera/releases/latest/download";
const RESTART_DELAY: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    All,
    Tera,
    Codex,
}

impl Component {
    fn tera(self) -> bool {
        matches!(self, Self::All | Self::Tera)
    }

    fn codex(self) -> bool {
        matches!(self, Self::All | Self::Codex)
    }
}

#[derive(Debug)]
pub struct UpdateOutcome {
    pub tera: BuildInfo,
    pub codex: Option<String>,
    pub restart_scheduled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Prepared,
    Installed,
    RolledBack { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpdateJournal {
    phase: Phase,
    previous: BuildInfo,
    next: BuildInfo,
    target: PathBuf,
    backup: Option<PathBuf>,
    codex_before: Option<String>,
    codex_after: Option<String>,
    codex_backup: Option<PathBuf>,
    codex_target: Option<PathBuf>,
}

struct CodexUpdate {
    before: String,
    after: String,
    backup: Option<PathBuf>,
    target: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum UpdateNotice {
    Applied {
        previous: BuildInfo,
        current: BuildInfo,
        codex_before: Option<String>,
        codex_after: Option<String>,
    },
    RolledBack {
        attempted: BuildInfo,
        restored: BuildInfo,
        reason: String,
        codex_before: Option<String>,
        codex_after: Option<String>,
    },
}

impl UpdateNotice {
    pub fn message(&self) -> String {
        match self {
            Self::Applied {
                previous,
                current,
                codex_before,
                codex_after,
            } => {
                let tera = if previous.commit_sha == current.commit_sha {
                    format!("Tera {} was already current", current.version)
                } else {
                    format!(
                        "Tera updated from {} ({}) to {} ({})",
                        previous.version,
                        previous.short_sha(),
                        current.version,
                        current.short_sha()
                    )
                };
                let codex = match (codex_before, codex_after) {
                    (Some(before), Some(after)) if before != after => {
                        format!(", and Codex updated from {before} to {after}")
                    }
                    (Some(after), _) | (_, Some(after)) => format!(", with {after}"),
                    _ => String::new(),
                };
                format!("{tera}{codex}. The new daemon passed its startup checks.")
            }
            Self::RolledBack {
                attempted,
                restored,
                reason,
                codex_before,
                codex_after,
            } => {
                if attempted.commit_sha == restored.commit_sha {
                    format!(
                        "The Codex update from {} to {} failed its first startup. Phoenix restored the previous Codex executable. {reason}",
                        codex_before.as_deref().unwrap_or("the prior version"),
                        codex_after.as_deref().unwrap_or("the new version")
                    )
                } else {
                    format!(
                        "The update to Tera {} ({}) failed its first startup. Phoenix restored {} ({}), including the prior Codex executable when it changed. {reason}",
                        attempted.version,
                        attempted.short_sha(),
                        restored.version,
                        restored.short_sha()
                    )
                }
            }
        }
    }
}

pub enum StartupAction {
    Continue(Option<Box<UpdateNotice>>),
    RestartAfterRollback,
}

struct PreparedUpdate {
    candidate: PathBuf,
    target: PathBuf,
    info: BuildInfo,
}

impl Drop for PreparedUpdate {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.candidate);
    }
}

pub fn run(config: &Config, component: Component, force: bool) -> Result<UpdateOutcome> {
    fs::create_dir_all(updates_dir(config))?;

    // Validate the Tera release before touching Codex. A missing or malformed
    // release must leave both installed programs exactly as they were.
    let prepared = if component.tera() {
        prepare_tera_update(force)?
    } else {
        None
    };

    let codex = if component.codex() {
        Some(update_codex(config)?)
    } else {
        None
    };
    let codex_before = codex.as_ref().map(|update| update.before.clone());
    let codex_after = codex.as_ref().map(|update| update.after.clone());

    let current = BuildInfo::current();
    if let Some(update) = codex.as_ref().filter(|update| update.backup.is_some()) {
        write_journal(
            config,
            &UpdateJournal {
                phase: Phase::Prepared,
                previous: current.clone(),
                next: current.clone(),
                target: current_executable()?,
                backup: None,
                codex_before: Some(update.before.clone()),
                codex_after: Some(update.after.clone()),
                codex_backup: update.backup.clone(),
                codex_target: update.target.clone(),
            },
        )?;
    }
    let tera_installed = prepared.is_some();
    let installed = match prepared {
        Some(prepared) => install_tera(config, prepared, codex.as_ref())?,
        None => current.clone(),
    };

    let codex_changed = codex_before != codex_after;
    let changed = tera_installed || codex_changed;
    if !tera_installed && codex_changed {
        write_journal(
            config,
            &UpdateJournal {
                phase: Phase::Installed,
                previous: current,
                next: installed.clone(),
                target: current_executable()?,
                backup: None,
                codex_before,
                codex_after: codex_after.clone(),
                codex_backup: codex.as_ref().and_then(|update| update.backup.clone()),
                codex_target: codex.as_ref().and_then(|update| update.target.clone()),
            },
        )?;
    } else if !tera_installed {
        if let Some(backup) = codex.as_ref().and_then(|update| update.backup.as_ref()) {
            let _ = fs::remove_file(backup);
        }
    }

    let restart_scheduled = if changed {
        schedule_daemon_restart(config, &current_executable()?)?
    } else {
        false
    };

    Ok(UpdateOutcome {
        tera: installed,
        codex: codex_after.or_else(codex_version),
        restart_scheduled,
    })
}

fn prepare_tera_update(force: bool) -> Result<Option<PreparedUpdate>> {
    let target = current_executable()?;
    let target_dir = target
        .parent()
        .ok_or_else(|| anyhow!("running executable has no parent: {target:?}"))?;
    let asset = release_asset()?;
    let base =
        std::env::var("TERA_UPDATE_BASE_URL").unwrap_or_else(|_| RELEASE_BASE_URL.to_string());
    let candidate = target_dir.join(format!(".tera-update-{}", std::process::id()));
    let checksum = target_dir.join(format!(".tera-update-{}.sha256", std::process::id()));

    let result = (|| -> Result<Option<PreparedUpdate>> {
        download(&format!("{base}/{asset}"), &candidate)?;
        download(&format!("{base}/{asset}.sha256"), &checksum)?;
        verify_checksum(&candidate, &checksum)?;

        let mut permissions = fs::metadata(&target)?.permissions();
        permissions.set_mode(permissions.mode() | 0o111);
        fs::set_permissions(&candidate, permissions)?;

        let info = binary_info(&candidate)?;
        let current = BuildInfo::current();
        if info.target != current.target {
            bail!(
                "release target {} does not match this binary's {}",
                info.target,
                current.target
            );
        }

        match compare_versions(&info.version, &current.version)? {
            std::cmp::Ordering::Less => bail!(
                "latest release {} is older than the installed {}",
                info.version,
                current.version
            ),
            std::cmp::Ordering::Equal if !force => {
                println!("Tera {} is already current.", current.version);
                return Ok(None);
            }
            _ => {}
        }

        Ok(Some(PreparedUpdate {
            candidate: candidate.clone(),
            target,
            info,
        }))
    })();

    let _ = fs::remove_file(&checksum);
    if result.is_err() {
        let _ = fs::remove_file(&candidate);
    }
    result
}

fn install_tera(
    config: &Config,
    mut prepared: PreparedUpdate,
    codex: Option<&CodexUpdate>,
) -> Result<BuildInfo> {
    let previous = BuildInfo::current();
    let backup = updates_dir(config).join("tera.previous");
    copy_atomic(&prepared.target, &backup)?;

    let mut journal = UpdateJournal {
        phase: Phase::Prepared,
        previous,
        next: prepared.info.clone(),
        target: prepared.target.clone(),
        backup: Some(backup),
        codex_before: codex.map(|update| update.before.clone()),
        codex_after: codex.map(|update| update.after.clone()),
        codex_backup: codex.and_then(|update| update.backup.clone()),
        codex_target: codex.and_then(|update| update.target.clone()),
    };
    write_journal(config, &journal)?;

    if let Err(error) = fs::rename(&prepared.candidate, &prepared.target) {
        let _ = restore(&journal);
        cleanup_update(config, &journal);
        return Err(error).context("failed to atomically install the Tera update");
    }
    sync_parent(&prepared.target);
    prepared.candidate.clear();

    match binary_info(&prepared.target) {
        Ok(installed) if installed.commit_sha == journal.next.commit_sha => {
            journal.phase = Phase::Installed;
            write_journal(config, &journal)?;
            Ok(installed)
        }
        result => {
            let reason = match result {
                Ok(installed) => format!(
                    "installed binary reports commit {}, expected {}",
                    installed.commit_sha, journal.next.commit_sha
                ),
                Err(error) => format!("installed binary could not report its version: {error:#}"),
            };
            restore(&journal)
                .context("the new binary failed validation and rollback also failed")?;
            journal.phase = Phase::RolledBack {
                reason: reason.clone(),
            };
            write_journal(config, &journal)?;
            bail!("the new binary failed validation and was rolled back: {reason}")
        }
    }
}

fn update_codex(config: &Config) -> Result<CodexUpdate> {
    let before = codex_version().ok_or_else(|| anyhow!("codex is not installed or not on PATH"))?;
    let before_target = executable_on_path("codex")?;
    let backup = updates_dir(config).join("codex.previous");
    copy_atomic(&before_target, &backup)?;
    let status = Command::new("codex")
        .arg("update")
        .status()
        .context("failed to start `codex update`")?;
    if !status.success() {
        restore_codex(&backup, &before_target, &before)?;
        let _ = fs::remove_file(&backup);
        bail!("`codex update` exited with {status}; Tera was not changed")
    }
    let after =
        codex_version().ok_or_else(|| anyhow!("Codex disappeared after its updater ran"))?;
    println!("Codex: {before} -> {after}");
    if before == after {
        let _ = fs::remove_file(&backup);
        return Ok(CodexUpdate {
            before,
            after,
            backup: None,
            target: None,
        });
    }
    Ok(CodexUpdate {
        before,
        after,
        backup: Some(backup),
        target: Some(executable_on_path("codex")?),
    })
}

pub fn startup_action(config: &Config, crashed: bool) -> Result<StartupAction> {
    let Some(mut journal) = read_journal(config)? else {
        return Ok(StartupAction::Continue(None));
    };
    let running = BuildInfo::current();

    if matches!(journal.phase, Phase::Prepared) {
        if running.commit_sha == journal.next.commit_sha {
            journal.phase = Phase::Installed;
            write_journal(config, &journal)?;
        } else if running.commit_sha == journal.previous.commit_sha {
            restore(&journal)?;
            cleanup_update(config, &journal);
            return Ok(StartupAction::Continue(None));
        }
    }

    match &journal.phase {
        Phase::Installed
            if crashed && (journal.backup.is_some() || journal.codex_backup.is_some()) =>
        {
            let reason = "The replacement did not reach a healthy daemon start.".to_string();
            restore(&journal)?;
            journal.phase = Phase::RolledBack { reason };
            write_journal(config, &journal)?;
            Ok(StartupAction::RestartAfterRollback)
        }
        Phase::Installed => Ok(StartupAction::Continue(Some(Box::new(
            UpdateNotice::Applied {
                previous: journal.previous,
                current: journal.next,
                codex_before: journal.codex_before,
                codex_after: journal.codex_after,
            },
        )))),
        Phase::RolledBack { reason } => Ok(StartupAction::Continue(Some(Box::new(
            UpdateNotice::RolledBack {
                attempted: journal.next.clone(),
                restored: journal.previous.clone(),
                reason: reason.clone(),
                codex_before: journal.codex_before.clone(),
                codex_after: journal.codex_after.clone(),
            },
        )))),
        Phase::Prepared => {
            warn!(
                "Ignoring an update journal that matches neither the running nor candidate build"
            );
            Ok(StartupAction::Continue(None))
        }
    }
}

/// Commit the update only after the daemon and its Codex app-server are ready.
pub fn mark_healthy(config: &Config) {
    let Ok(Some(journal)) = read_journal(config) else {
        return;
    };
    let running_sha = BuildInfo::current().commit_sha;
    let expected = match journal.phase {
        Phase::Installed => journal.next.commit_sha.as_str(),
        Phase::RolledBack { .. } => journal.previous.commit_sha.as_str(),
        Phase::Prepared => return,
    };
    if running_sha == expected {
        cleanup_update(config, &journal);
        info!("Committed the healthy update and removed its rollback copy");
    }
}

fn restore(journal: &UpdateJournal) -> Result<()> {
    if let Some(backup) = &journal.backup {
        let backup_info = binary_info(backup)?;
        if backup_info.commit_sha != journal.previous.commit_sha {
            bail!(
                "rollback copy is {}, expected {}",
                backup_info.commit_sha,
                journal.previous.commit_sha
            );
        }
        copy_atomic(backup, &journal.target)?;
        let restored = binary_info(&journal.target)?;
        if restored.commit_sha != journal.previous.commit_sha {
            bail!("rollback binary did not survive installation validation")
        }
    }

    if let (Some(backup), Some(target), Some(before)) = (
        &journal.codex_backup,
        &journal.codex_target,
        &journal.codex_before,
    ) {
        restore_codex(backup, target, before)?;
    }
    Ok(())
}

fn restore_codex(backup: &Path, target: &Path, expected_version: &str) -> Result<()> {
    copy_atomic(backup, target).context("failed to restore the previous Codex executable")?;
    let restored = Command::new(target)
        .arg("--version")
        .output()
        .context("restored Codex could not report its version")?;
    let version = String::from_utf8_lossy(&restored.stdout).trim().to_string();
    if !restored.status.success() || version != expected_version {
        bail!("Codex rollback reports {version:?}, expected {expected_version:?}")
    }
    Ok(())
}

fn executable_on_path(name: &str) -> Result<PathBuf> {
    let path = std::env::var_os("PATH")
        .and_then(|path| {
            std::env::split_paths(&path)
                .map(|directory| directory.join(name))
                .find(|candidate| candidate.is_file())
        })
        .ok_or_else(|| anyhow!("{name} is not on PATH"))?;
    path.canonicalize()
        .with_context(|| format!("could not resolve {}", path.display()))
}

fn binary_info(path: &Path) -> Result<BuildInfo> {
    let output = Command::new(path)
        .args(["version", "--json"])
        .output()
        .with_context(|| format!("could not run {}", path.display()))?;
    if !output.status.success() {
        bail!(
            "{} version check exited with {}",
            path.display(),
            output.status
        )
    }
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("{} returned invalid version JSON", path.display()))
}

fn release_asset() -> Result<String> {
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("x86_64", "linux") => Ok("tera-x86_64-unknown-linux-gnu".to_string()),
        ("aarch64", "macos") => Ok("tera-aarch64-apple-darwin".to_string()),
        (arch, os) => bail!("automatic Tera updates are not published for {arch}-{os}"),
    }
}

fn download(url: &str, destination: &Path) -> Result<()> {
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(destination)
        .arg(url)
        .status()
        .with_context(|| "curl is required for automatic Tera updates")?;
    if !status.success() {
        bail!("download failed for {url}")
    }
    Ok(())
}

fn verify_checksum(candidate: &Path, checksum_file: &Path) -> Result<()> {
    let raw = fs::read_to_string(checksum_file)?;
    let expected = raw
        .split_whitespace()
        .next()
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| anyhow!("release checksum is malformed"))?;

    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(candidate)
        .output()
        .or_else(|_| Command::new("sha256sum").arg(candidate).output())
        .context("shasum or sha256sum is required to verify the update")?;
    if !output.status.success() {
        bail!("could not calculate the update checksum")
    }
    let actual = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if !actual.eq_ignore_ascii_case(expected) {
        bail!("update checksum mismatch; the downloaded binary was not installed")
    }
    Ok(())
}

fn compare_versions(left: &str, right: &str) -> Result<std::cmp::Ordering> {
    fn parse(value: &str) -> Result<Vec<u64>> {
        value
            .trim_start_matches('v')
            .split_once('-')
            .map_or(value.trim_start_matches('v'), |(core, _)| core)
            .split('.')
            .map(|part| {
                part.parse::<u64>()
                    .with_context(|| format!("invalid release version {value:?}"))
            })
            .collect()
    }

    Ok(parse(left)?.cmp(&parse(right)?))
}

fn current_executable() -> Result<PathBuf> {
    std::env::current_exe()
        .context("could not locate the running Tera binary")?
        .canonicalize()
        .context("could not resolve the running Tera binary")
}

fn updates_dir(config: &Config) -> PathBuf {
    config.runtime_dir().join("updates")
}

fn journal_path(config: &Config) -> PathBuf {
    updates_dir(config).join("update.json")
}

fn read_journal(config: &Config) -> Result<Option<UpdateJournal>> {
    let path = journal_path(config);
    let mut raw = String::new();
    match File::open(&path) {
        Ok(mut file) => file.read_to_string(&mut raw)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    serde_json::from_str(&raw)
        .with_context(|| format!("update journal {} is corrupt", path.display()))
        .map(Some)
}

fn write_journal(config: &Config, journal: &UpdateJournal) -> Result<()> {
    let mut serialized = serde_json::to_vec_pretty(journal)?;
    serialized.push(b'\n');
    crate::runtime::write_atomic(&journal_path(config), &serialized, 0o600)
}

fn copy_atomic(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("destination has no parent: {destination:?}"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}-{}.tmp",
        destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        std::process::id()
    ));
    fs::copy(source, &temporary)?;
    fs::set_permissions(&temporary, fs::metadata(source)?.permissions())?;
    File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, destination)?;
    sync_parent(destination);
    Ok(())
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
}

fn cleanup_update(config: &Config, journal: &UpdateJournal) {
    if let Some(backup) = &journal.backup {
        let _ = fs::remove_file(backup);
    }
    if let Some(backup) = &journal.codex_backup {
        let _ = fs::remove_file(backup);
    }
    let _ = fs::remove_file(journal_path(config));
}

fn schedule_daemon_restart(config: &Config, binary: &Path) -> Result<bool> {
    let Some(pid) = locked_daemon_pid(&config.lock_file_path())? else {
        println!("No running daemon was found. The new version will be used next time it starts.");
        return Ok(false);
    };
    Command::new(binary)
        .arg("__signal-update")
        .arg("--lock")
        .arg(config.lock_file_path())
        .arg("--pid")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("could not schedule the daemon restart")?;
    Ok(true)
}

fn locked_daemon_pid(lock_path: &Path) -> Result<Option<u32>> {
    let mut file = match OpenOptions::new().read(true).write(true).open(lock_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let locked = unsafe {
        libc::flock(
            std::os::fd::AsRawFd::as_raw_fd(&file),
            libc::LOCK_EX | libc::LOCK_NB,
        )
    };
    if locked == 0 {
        let _ = unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_UN) };
        return Ok(None);
    }
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;
    Ok(raw.trim().parse().ok())
}

pub fn signal_update(lock_path: &Path, pid: u32) -> Result<()> {
    thread::sleep(RESTART_DELAY);
    if locked_daemon_pid(lock_path)? != Some(pid) {
        return Ok(());
    }
    let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGUSR1) };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).context("could not signal the daemon");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executable(path: &Path, info: &BuildInfo) {
        let json = serde_json::to_string(info).unwrap();
        fs::write(path, format!("#!/bin/sh\nprintf '%s\\n' '{json}'\n")).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    #[test]
    fn test_version_comparison_is_numeric() {
        assert_eq!(
            compare_versions("0.10.0", "0.9.9").unwrap(),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            compare_versions("v1.2.3", "1.2.3").unwrap(),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_versions("1.2.3-rc.1", "1.2.4").unwrap(),
            std::cmp::Ordering::Less
        );
        assert!(compare_versions("latest", "1.0.0").is_err());
    }

    #[test]
    fn test_update_notice_names_the_installed_build() {
        let mut previous = BuildInfo::current();
        previous.version = "1.0.0".to_string();
        previous.commit_sha = "1111111111111111".to_string();
        let mut current = previous.clone();
        current.version = "1.1.0".to_string();
        current.commit_sha = "2222222222222222".to_string();
        let message = UpdateNotice::Applied {
            previous,
            current,
            codex_before: Some("codex-cli 1.0".to_string()),
            codex_after: Some("codex-cli 1.1".to_string()),
        }
        .message();
        assert!(message.contains("1.0.0 (111111111111)"));
        assert!(message.contains("1.1.0 (222222222222)"));
        assert!(message.contains("codex-cli 1.1"));
    }

    #[test]
    fn test_a_crashed_first_start_restores_the_previous_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().join("workspace"), true);
        let target = tmp.path().join("tera");
        let backup = tmp.path().join("tera.previous");
        let previous = BuildInfo::current();
        let mut next = previous.clone();
        next.version = "99.0.0".to_string();
        next.commit_sha = "9999999999999999999999999999999999999999".to_string();
        executable(&target, &next);
        executable(&backup, &previous);

        write_journal(
            &config,
            &UpdateJournal {
                phase: Phase::Installed,
                previous: previous.clone(),
                next,
                target: target.clone(),
                backup: Some(backup),
                codex_before: None,
                codex_after: None,
                codex_backup: None,
                codex_target: None,
            },
        )
        .unwrap();

        assert!(matches!(
            startup_action(&config, true).unwrap(),
            StartupAction::RestartAfterRollback
        ));
        assert_eq!(binary_info(&target).unwrap(), previous);
        assert!(matches!(
            read_journal(&config).unwrap().unwrap().phase,
            Phase::RolledBack { .. }
        ));
    }

    #[test]
    fn test_a_healthy_start_removes_only_update_owned_state() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().join("workspace"), true);
        let backup = tmp.path().join("tera.previous");
        fs::write(&backup, "rollback copy").unwrap();
        let current = BuildInfo::current();
        write_journal(
            &config,
            &UpdateJournal {
                phase: Phase::Installed,
                previous: current.clone(),
                next: current,
                target: tmp.path().join("tera"),
                backup: Some(backup.clone()),
                codex_before: None,
                codex_after: None,
                codex_backup: None,
                codex_target: None,
            },
        )
        .unwrap();
        let unrelated = updates_dir(&config).join("keep-me");
        fs::write(&unrelated, "mine").unwrap();

        mark_healthy(&config);

        assert!(!journal_path(&config).exists());
        assert!(!backup.exists());
        assert!(unrelated.exists());
    }

    #[test]
    fn test_a_crashed_codex_only_update_restores_codex() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::new(tmp.path().join("workspace"), true);
        let target = tmp.path().join("codex");
        let backup = tmp.path().join("codex.previous");
        fs::write(&target, "#!/bin/sh\necho 'codex-cli 2.0'\n").unwrap();
        fs::write(&backup, "#!/bin/sh\necho 'codex-cli 1.0'\n").unwrap();
        for path in [&target, &backup] {
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).unwrap();
        }
        let current = BuildInfo::current();
        write_journal(
            &config,
            &UpdateJournal {
                phase: Phase::Installed,
                previous: current.clone(),
                next: current,
                target: tmp.path().join("tera"),
                backup: None,
                codex_before: Some("codex-cli 1.0".to_string()),
                codex_after: Some("codex-cli 2.0".to_string()),
                codex_backup: Some(backup),
                codex_target: Some(target.clone()),
            },
        )
        .unwrap();

        assert!(matches!(
            startup_action(&config, true).unwrap(),
            StartupAction::RestartAfterRollback
        ));
        let output = Command::new(target).arg("--version").output().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "codex-cli 1.0"
        );
    }
}
