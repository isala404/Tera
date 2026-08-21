//! Evidence that the previous process did not get to say goodbye.
//!
//! A file, not a row in the runtime database. The database is one of the things
//! a crash can take with it, and the whole point of this record is to still be
//! readable when the rest of the system is not.
//!
//! Armed at startup and removed on a clean exit, so its mere presence at boot is
//! the signal. That one test covers panics, OOM kills, `SIGKILL` and host
//! reboots without having to enumerate them.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Where the panic hook writes. Set once, when the mark is armed.
static MARK_PATH: OnceLock<PathBuf> = OnceLock::new();

/// What a process leaves behind while it runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrashMark {
    pub started_at_ms: i64,
    /// Filled in by the panic hook. Absent when the process died without
    /// running any Rust code on the way out, which is most kills.
    pub panic: Option<String>,
    /// Unclean starts in a row. Reset by any clean exit, and what tells Phoenix
    /// it is in a crash loop rather than handling a one-off.
    #[serde(default)]
    pub consecutive: u32,
}

impl CrashMark {
    /// A sentence for the owner. This is the "what happened" half of Phoenix, so
    /// it says what is actually known rather than a generic apology.
    pub fn describe(&self) -> String {
        match &self.panic {
            Some(panic) => format!("crashed: {panic}"),
            None => "stopped without shutting down cleanly".to_string(),
        }
    }
}

/// Read any mark the last process left, then arm one for this process.
///
/// Returns the previous life, if there was an unclean one. The panic hook is
/// installed here so a panic anywhere records its cause before we die; a panic
/// in a spawned task that the runtime absorbs is recorded too, but a clean exit
/// removes the file, so it is only ever reported when the process really died.
pub fn arm(runtime_dir: &Path) -> Result<Option<CrashMark>> {
    let path = runtime_dir.join("phoenix.json");
    let prior = read(&path);

    std::fs::create_dir_all(runtime_dir)?;
    write(
        &path,
        &CrashMark {
            started_at_ms: chrono::Utc::now().timestamp_millis(),
            panic: None,
            // Carried forward, not restarted: a clean exit is what clears it.
            consecutive: prior.as_ref().map_or(0, |p| p.consecutive + 1),
        },
    )?;

    if MARK_PATH.set(path).is_ok() {
        install_panic_hook();
    }

    Ok(prior)
}

/// Remove the mark. Anything that skips this is, by definition, a crash.
pub fn disarm(runtime_dir: &Path) {
    let path = runtime_dir.join("phoenix.json");
    if let Err(e) = std::fs::remove_file(&path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!("Could not clear the crash mark at {path:?}: {e}");
        }
    }
}

fn read(path: &Path) -> Option<CrashMark> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write(path: &Path, mark: &CrashMark) -> Result<()> {
    std::fs::write(path, serde_json::to_string(mark)?)?;
    Ok(())
}

/// Record the panic message and location into the mark, then defer to whatever
/// hook was already installed so the normal backtrace still reaches the log.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(path) = MARK_PATH.get() {
            let where_ = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "unknown location".to_string());
            let mut mark = read(path).unwrap_or_default();
            mark.panic = Some(format!("{info} (at {where_})"));
            let _ = write(path, &mark);
        }
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_an_armed_mark_is_reported_to_the_next_start_and_a_disarmed_one_is_not() {
        let dir = tempfile::tempdir().unwrap();

        assert!(arm(dir.path()).unwrap().is_none(), "a first start is not a crash");

        // Simulating a crash means simply not disarming.
        let prior = arm(dir.path()).unwrap().expect("an armed mark survives");
        assert!(prior.started_at_ms > 0);
        assert_eq!(prior.consecutive, 0);
        assert_eq!(prior.describe(), "stopped without shutting down cleanly");

        // Crashing again is what a crash loop looks like from here.
        assert_eq!(arm(dir.path()).unwrap().unwrap().consecutive, 1);

        disarm(dir.path());
        assert!(arm(dir.path()).unwrap().is_none(), "a clean exit leaves nothing");
    }

    #[test]
    fn test_a_recorded_panic_becomes_the_reported_cause() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("phoenix.json");
        write(
            &path,
            &CrashMark {
                started_at_ms: 1,
                panic: Some("index out of bounds (at src/foo.rs:12)".to_string()),
                consecutive: 0,
            },
        )
        .unwrap();

        let prior = arm(dir.path()).unwrap().unwrap();
        assert_eq!(
            prior.describe(),
            "crashed: index out of bounds (at src/foo.rs:12)"
        );
    }
}
